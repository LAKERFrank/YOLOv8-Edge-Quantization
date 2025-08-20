import json
from pathlib import Path
import threading
import time
import queue
from typing import Tuple, List

import cv2
import numpy as np
import torch
import paho.mqtt.client as mqtt

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.tracknet.protocal.point import Point
from ultralytics.tracknet.utils.nms import non_max_suppression
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.tracknet.protocal.image_buffer import FrameProtocol, ImageBufferProtocol

class ImageBufferPredictor:
    def __init__(self, weight: str, image_buffer: ImageBufferProtocol, output_width: int = None,
                 output_height: int = None, mqttc: mqtt.Client = None, output_topic: str = None,
                 cfg=DEFAULT_CFG, overrides=None):

        self.args = get_cfg(cfg, overrides)
        # self.save_dir = self.get_save_dir()
        self.model = AutoBackend(weight,
                                 device=select_device(self.args.device, verbose=False),
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=False)
        self.device = self.model.device
        self.args.half = self.model.fp16
        self.model.eval()

        self.output_width = output_width
        self.output_height = output_height
        self.mqttc = mqttc
        self.output_topic = output_topic
        self.image_buffer = image_buffer
        self.track_size = 10
        self.imgsz = 640

        self.max_streams = torch.cuda.device_count() * 2
        self.streams = [torch.cuda.Stream(device=self.device) for _ in range(self.max_streams)]
        self.event_pool = queue.SimpleQueue()

        for _ in range(128):
            self.event_pool.put(torch.cuda.Event())

        self.stream_idx = 0
        self.infer_q = queue.Queue(maxsize=128)
        self.result_q = queue.Queue(maxsize=256)
        self._stopper = threading.Event()


    def get_save_dir(self):
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        return increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
    
    def start_preprocess(self):
        self.running = True
        self.preprocess_thread = threading.Thread(target=self._preprocess_loop)
        self.preprocess_thread.start()

    def start_for_test(self):
        self.running = True
        self.threads = [
            threading.Thread(target=self._inference_loop),
            threading.Thread(target=self._postprocess_loop),
        ]
        for t in self.threads:
            t.start()
        for t in self.threads:
            t.join()  # 阻塞直到所有 thread 結束

    def start(self):
        print("[Predictor] Start")
        self.running = True
        self.threads = [
            threading.Thread(target=self._preprocess_loop),
            threading.Thread(target=self._inference_loop),
            threading.Thread(target=self._postprocess_loop),
        ]
        for t in self.threads:
            t.start()
        for t in self.threads:
            t.join()  # 阻塞直到所有 thread 結束
        print("[Predictor] All threads finished.")

    def stop(self):
        self.running = False

    def _preprocess_loop(self):
        while self.running:
            try:
                tensor, fids, timestamps = self._preprocess()
                self.stream_idx = (self.stream_idx + 1) % self.max_streams
                self.infer_q.put((tensor, (fids, timestamps), self.stream_idx), timeout=1)
                if not self.running:
                    self.infer_q.put(None)
            except Exception as e:
                LOGGER.warning(f"Preprocess loop error: {e}")

    def _preprocess(self) -> Tuple[torch.Tensor, List[int], List[float]]:
        frames, fids, timestamps = [], [], []
        while len(frames) < self.track_size:
            frame = self.image_buffer.pop(True)

            img = frame.image.astype(np.float32)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = self.pad_to_square(img)
            img = cv2.resize(img, dsize=(self.imgsz, self.imgsz), interpolation=cv2.INTER_CUBIC)
            frames.append(np.expand_dims(img, axis=0))
            fids.append(frame.index)
            timestamps.append(frame.monotonic_timestamp)
            if frame.is_eos:
                self.stop()
                break

        if len(frames) < self.track_size:
            for _ in range(self.track_size - len(frames)):
                img = np.zeros((self.imgsz, self.imgsz), dtype=np.float32)
                frames.append(np.expand_dims(img, axis=0))
                fids.append(-1)
                timestamps.append(-1)
                
        img = np.concatenate(frames, 0)
        return torch.from_numpy(img).contiguous().pin_memory(), fids, timestamps

    def _inference_loop(self):
        while self.running:
            try:
                item = self.infer_q.get(timeout=0.1)
                if item is None:
                    self.result_q.put(None)
                    break

                tensor, meta, stream_id = item
                stream = self.streams[stream_id]
                event = self.event_pool.get()

                with torch.cuda.stream(stream):
                    input_gpu = tensor.to(self.device, non_blocking=True)
                    median = input_gpu.median(dim=0).values
                    input_gpu.sub_(median).clamp_(0, 255).div_(255.0)
                    input_gpu = input_gpu.unsqueeze(0)
                    if self.model.fp16:
                        input_gpu = input_gpu.half()
                    with torch.no_grad():
                        output = self.model(input_gpu)
                    event.record(stream)

                self.result_q.put((event, output, meta, stream))
            except queue.Empty:
                continue
            except Exception as e:
                LOGGER.warning(f"Inference loop error: {e}")

    def _postprocess_loop(self):
        while self.running:
            try:
                item = self.result_q.get(timeout=0.1)
                if item is None:
                    break
                event, output, meta, stream = item
                if event.query():
                    self._postprocess(output, meta)
                    self.event_pool.put(event)
                else:
                    # 沒完成的重新放回，但避免 busy loop
                    self.result_q.put((event, output, meta, stream))
                    time.sleep(0.001)
            except queue.Empty:
                time.sleep(0.001)
            except Exception as e:
                LOGGER.warning(f"Postprocess loop error: {e}")
                raise e


    def pad_to_square(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        size = max(h, w)
        padded = np.zeros((size, size), dtype=img.dtype)
        padded[:h, :w] = img
        return padded

    def _postprocess(self, output_tensor: torch.Tensor, meta:Tuple[List[int], List[float]]):
        fids, timestamps = meta

        use_nms = True
        conf_threshold = 0.5
        nc = 1
        feat_no = 8
        cell_num = 80
        stride = 8

        feats = output_tensor[0][0]
        pred_distri, pred_probs = feats.view(feat_no + nc, -1).split(
            (feat_no, nc), 0)
        
        pred_probs = pred_probs.permute(1, 0)
        pred_pos = pred_distri.permute(1, 0)

        each_probs = pred_probs.view(10, cell_num, cell_num)
        each_pos_x, each_pos_y, each_pos_nx, each_pos_ny = pred_pos.view(10, cell_num, cell_num, feat_no).split([2, 2, 2, 2], dim=3)

        frame_preds = []
        metadata = []
        for frame_idx in range(10):
            p_cell_x = each_pos_x[frame_idx]
            p_cell_y = each_pos_y[frame_idx]
            p_cell_nx = each_pos_nx[frame_idx]
            p_cell_ny = each_pos_ny[frame_idx]
            fid = fids[frame_idx]
            timestamp = timestamps[frame_idx]
            center = 0.5

            # 獲取當前圖片的 conf
            p_conf = each_probs[frame_idx]

            if use_nms:
                nms_preds = non_max_suppression(p_conf, p_cell_x, p_cell_y, conf_threshold=conf_threshold, dis_tolerance=20)

                # 取出 nms 的結果
                for pred in nms_preds:
                    max_x, max_y, max_conf = pred
                    pred_x = max_x*stride + (center*stride-p_cell_x[int(max_y)][int(max_x)][0]+p_cell_x[int(max_y)][int(max_x)][1])
                    pred_y = max_y*stride + (center*stride-p_cell_y[int(max_y)][int(max_x)][0]+p_cell_y[int(max_y)][int(max_x)][1])

                    if fid != -1:
                        frame_preds.append((pred_x, pred_y, max_conf))
                        metadata.append((fid, timestamp))
            else:
                p_conf_masked = p_conf * (p_conf >= conf_threshold).float()
                max_position = torch.argmax(p_conf_masked)
                # max_y, max_x = np.unravel_index(max_position, p_conf.shape)
                max_y, max_x = np.unravel_index(max_position.cpu().numpy(), p_conf.shape)
                max_conf = p_conf[max_y, max_x].item()

                pred_x = max_x*stride + (center*stride-p_cell_x[max_y][max_x][0]+p_cell_x[max_y][max_x][1])
                pred_y = max_y*stride + (center*stride-p_cell_y[max_y][max_x][0]+p_cell_y[max_y][max_x][1])
                if fid != -1:
                    frame_preds.append((pred_x, pred_y, max_conf))
                    metadata.append((fid, timestamp))
        result = (frame_preds, metadata)
        if self.mqttc is not None:
            self._publishPoints((frame_preds, metadata) if use_nms else (frame_preds[:1], metadata[:1]))
        # print("[Result] output shape:", len(frame_preds), "fid", fid, "timestamp", timestamp, "endTime", time.monotonic())
        return result

    def _publishPoints(self, resultItems):
        (frame_preds, metadata) = resultItems
        points = []
        for i in range(len(frame_preds)):
            (output_x, output_y, output_conf) = frame_preds[i]
            (fid, timestamp) = metadata[i]
            output_x = output_x.item()
            output_y = output_y.item()

            points.append(Point(
                fid=fid,
                timestamp=timestamp,
                visibility=output_conf>=0.5,
                x=output_x,
                y=output_y,
                ))
        payload = {"linear": [p.toJson() for p in points]}
        self.mqttc.publish(self.output_topic, json.dumps(payload))
