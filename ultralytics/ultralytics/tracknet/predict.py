
import csv
from datetime import datetime
import json
import os
from queue import Empty, Queue
import threading
from matplotlib import pyplot as plt
import pandas as pd
import torch
import numpy as np
from ultralytics.tracknet.pred_dataset import TrackNetPredDataset
from ultralytics.tracknet.protocal.point import Point
from ultralytics.tracknet.utils.nms import non_max_suppression
from ultralytics.yolo.data.build import load_inference_source
from ultralytics.yolo.engine.predictor import STREAM_WARNING, BasePredictor
from ultralytics.yolo.engine.results import Results
from torch.utils.data import DataLoader

from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode
from pathlib import Path
import cv2
from dataclasses import dataclass
import time
import torch.profiler
import threading
import queue as Q
from contextlib import contextmanager
import paho.mqtt.client as mqtt
from torch.utils.data import Dataset

@dataclass
class Prediction:
    x: float
    y: float
    conf: float

@dataclass
class ResultItem:
    pred: Prediction
    speed: dict[str, float | None]

class TrackNetPredictor(BasePredictor):
    def __init__(self, output_width:int=None, output_height:int=None,
                 mqttc:mqtt.Client=None, output_topic:str=None, dataset:Dataset = None,
                 cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.output_width = output_width
        self.output_height = output_height
        self.mqttc = mqttc
        self.output_topic = output_topic
        self.dataset = dataset

    # def profile_resources(self, tag=""):
    #     cpu = self.proc.cpu_percent(interval=None)
    #     mem = self.proc.memory_info().rss / 1024**2
    #     if self.gpu_available:
    #         torch.cuda.synchronize()
    #         util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
    #         mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
    #         print(f"[{tag}] CPU: {cpu:.1f}%, RAM: {mem:.1f}MB, GPU: {util.gpu}%, vRAM: {mem_info.used/1024**2:.1f}MB")
    #     else:
    #         print(f"[{tag}] CPU: {cpu:.1f}%, RAM: {mem:.1f}MB (No GPU available)")

    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = None
        self.dataset = self.dataset if self.dataset else TrackNetPredDataset(
            dir=source,
            num_input=10,
            imgsz=640,
        )
        # self.source_type = self.dataset.source_type
        # self.vid_path, self.vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs
    # def setup_model(self, model, verbose=True):
    #     """Initialize YOLO model with given parameters and set it to evaluation mode."""
    #     self.model = model
    #     self.device = select_device(self.args.device, verbose=verbose)  # update device
    #     self.args.half = True  # update half
    #     # self.args.half = self.args.half  # update half
    #     self.model.eval()
    def cpu_preprocess(self, im0s):
        """CPU version of preprocess"""
        if not isinstance(im0s, torch.Tensor):
            im = torch.from_numpy(im0s).permute(2, 0, 1).contiguous().float()  # (HWC) -> (CHW)
        else:
            im = im0s.float()

        # 一律留在 CPU上處理
        median = im.median(dim=0).values  # (H, W)
        im = (im - median.unsqueeze(0)).clamp(0, 255) / 255.0

        # 注意：此時 im shape = (C, H, W)
        return im
    def preprocess(self, im):
        not_tensor = not isinstance(im, torch.Tensor)

        if not_tensor:
            im = torch.from_numpy(im)    # 直接轉換, shape = (H, W, C)
            im = im.permute(2, 0, 1).contiguous()  # (HWC -> CHW)

        im = im.to(self.device, dtype=torch.float32, non_blocking=True)
        median = im.median(dim=1).values  # shape: (H, W)
        im.sub_(median).div_(255.0)

        # im = im.unsqueeze(0)

        if self.model.fp16:
            im = im.half()

        return im
    
        # assert im.ndim == 3 and im.shape[0] == 10, "Expect shape (10, H, W)"
        # img = im.to(self.device)

        # img = img.float()  # 若 im 是 uint8，轉為 float32
        # median = img.median(dim=0).values  # shape: (H, W)
        # img = img - median
        # img = torch.clamp(img, 0, 255)

        # # Normalize
        # img /= 255.0

        # # Output shape: (1, 10, 640, 640)
        # result = img.unsqueeze(0).to(self.device).half() if self.model.fp16 else img.unsqueeze(0).to(self.device)
        # # self.profile_resources("Preprocess (after)")
        # return result

    # def inference(self, im, *args, **kwargs):
    #     self.profile_resources("Inference (before)")
    #     with torch.profiler.profile(
    #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #         record_shapes=True
    #     ) as prof:
    #         result = super().inference(im, *args, **kwargs)
    #         torch.cuda.synchronize()
    #         prof.step()
            
    #     print(prof.key_averages().table(sort_by="cuda_time_total"))
    #     self.profile_resources("Inference (after)")
    #     return result

    def preprocess_with_log(self, im):
        timings = {}

        # Step 1: 檢查是否為 tensor
        t0 = time.perf_counter()
        not_tensor = not isinstance(im, torch.Tensor)
        t1 = time.perf_counter()
        timings["is_tensor_check"] = (t1 - t0) * 1000

        # Step 2: 若是 numpy，轉成 torch tensor
        if not_tensor:
            t2 = time.perf_counter()
            im = torch.from_numpy(im)    # 直接轉換, shape = (H, W, C)
            im = im.permute(2, 0, 1).contiguous()  # (HWC -> CHW)
            t3 = time.perf_counter()
            timings["numpy_to_tensor"] = (t3 - t2) * 1000

        # Step 3: 移動到 device，並轉 float32
        t4 = time.perf_counter()
        im = im.to(self.device, dtype=torch.float32, non_blocking=self.device.type == "cuda")
        t5 = time.perf_counter()
        timings["to_device_and_fp32"] = (t5 - t4) * 1000

        # Step 4: median subtraction
        t6 = time.perf_counter()
        median = im.median(dim=0).values  # shape: (H, W)
        im.sub_(median)
        t7 = time.perf_counter()
        timings["median_subtract"] = (t7 - t6) * 1000

        # Step 5: clamp & normalize
        t8 = time.perf_counter()
        im.clamp_(0, 255).div_(255.0)
        t9 = time.perf_counter()
        timings["clamp_and_normalize"] = (t9 - t8) * 1000

        # Step 6: add batch dim
        t10 = time.perf_counter()
        im = im.unsqueeze(0)
        t11 = time.perf_counter()
        timings["unsqueeze"] = (t11 - t10) * 1000

        # Step 7: convert to half if needed
        t12 = time.perf_counter()
        if self.model.fp16:
            im = im.half()
        t13 = time.perf_counter()
        timings["fp16_convert"] = (t13 - t12) * 1000

        # Log all timing
        print("[Preprocess Timing (ms)]")
        for k, v in timings.items():
            print(f"{k:25}: {v:.3f} ms")

        return im

    def cpu_postprocess(self, preds, im0s):
        """CPU version of postprocess"""
        use_nms = True
        conf_threshold = 0.5
        nc = 1
        reg_max = 16
        feat_no = 8
        no = nc + reg_max * feat_no
        cell_num = 80
        stride = 8
        proj = torch.arange(reg_max, dtype=torch.float)

        # 把 preds 搬回 CPU
        feats = preds[0].detach().cpu()

        pred_distri, pred_scores = feats.view(no, -1).split((reg_max * feat_no, nc), 0)
        pred_scores = pred_scores.permute(1, 0).contiguous()
        pred_distri = pred_distri.permute(1, 0).contiguous()

        pred_probs = torch.sigmoid(pred_scores)
        a, c = pred_distri.shape

        pred_pos = pred_distri.view(a, feat_no, c // feat_no).softmax(2).matmul(proj.type(pred_distri.dtype))
        each_probs = pred_probs.view(10, cell_num, cell_num)
        each_pos_x, each_pos_y, each_pos_nx, each_pos_ny = pred_pos.view(10, cell_num, cell_num, feat_no).split([2, 2, 2, 2], dim=3)

        result = []
        for frame_idx in range(10):
            p_cell_x = each_pos_x[frame_idx]
            p_cell_y = each_pos_y[frame_idx]
            p_cell_nx = each_pos_nx[frame_idx]
            p_cell_ny = each_pos_ny[frame_idx]
            center = 0.5

            p_conf = each_probs[frame_idx]
            frame_preds = []

            if use_nms:
                nms_preds = non_max_suppression(p_conf, p_cell_x, p_cell_y, conf_threshold=conf_threshold, dis_tolerance=20)

                for pred in nms_preds:
                    max_x, max_y, max_conf = pred
                    pred_x = max_x*stride + (center*stride-p_cell_x[int(max_y)][int(max_x)][0]+p_cell_x[int(max_y)][int(max_x)][1])
                    pred_y = max_y*stride + (center*stride-p_cell_y[int(max_y)][int(max_x)][0]+p_cell_y[int(max_y)][int(max_x)][1])

                    frame_preds.append(ResultItem(
                        pred=Prediction(x=pred_x, y=pred_y, conf=max_conf),
                        speed={'preprocess': None, 'inference': None, 'postprocess': None}
                    ))
            else:
                p_conf_masked = p_conf * (p_conf >= conf_threshold).float()
                max_position = torch.argmax(p_conf_masked)
                max_y, max_x = np.unravel_index(max_position.cpu().numpy(), p_conf.shape)
                max_conf = p_conf[max_y, max_x].item()

                pred_x = max_x*stride + (center*stride-p_cell_x[max_y][max_x][0]+p_cell_x[max_y][max_x][1])
                pred_y = max_y*stride + (center*stride-p_cell_y[max_y][max_x][0]+p_cell_y[max_y][max_x][1])
                frame_preds.append(ResultItem(
                    pred=Prediction(x=pred_x, y=pred_y, conf=max_conf),
                    speed={'preprocess': None, 'inference': None, 'postprocess': None}
                ))

            result.append(ResultItem(
                pred=frame_preds if use_nms else frame_preds[0],
                speed={'preprocess': None, 'inference': None, 'postprocess': None}
            ))

        return result

    # postprocess_output_memory
    def postprocess(self, preds, img, orig_imgs, fids, timestamps):
        """Postprocesses predictions and returns a list of Results objects."""
        # self.profile_resources("Postprocess (before)")
        use_nms = True
        conf_threshold = 0.5
        nc = 1
        feat_no = 8
        cell_num = 80
        stride = 8

        feats = preds[0][0]
        pred_distri, pred_probs = feats.view(feat_no + nc, -1).split(
            (feat_no, nc), 0)
        
        pred_probs = pred_probs.permute(1, 0)
        pred_pos = pred_distri.permute(1, 0)

        each_probs = pred_probs.view(10, cell_num, cell_num)
        each_pos_x, each_pos_y, each_pos_nx, each_pos_ny = pred_pos.view(10, cell_num, cell_num, feat_no).split([2, 2, 2, 2], dim=3)

        result = []
        orig_images_clone = orig_imgs.squeeze(0).contiguous().cpu().numpy()

        p = Path(self.batch[0][0])
        parent_dir = p.parent.name
        match_dir = p.parent.parent.parent.name
        frame_save_path = os.path.join(self.save_dir, match_dir, 'frame', parent_dir)
        csv_save_path = os.path.join(self.save_dir, match_dir, 'csv', parent_dir)
        os.makedirs(frame_save_path, exist_ok=True)
        os.makedirs(csv_save_path, exist_ok=True)
        csv_rows = []
        real_frame_idx = int(p.stem)

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

            frame_preds = []
            metadata = []
            if use_nms:
                nms_preds = non_max_suppression(p_conf, p_cell_x, p_cell_y, conf_threshold=conf_threshold, dis_tolerance=20)

                # 取出 nms 的結果
                for pred in nms_preds:
                    max_x, max_y, max_conf = pred
                    pred_x = max_x*stride + (center*stride-p_cell_x[int(max_y)][int(max_x)][0]+p_cell_x[int(max_y)][int(max_x)][1])
                    pred_y = max_y*stride + (center*stride-p_cell_y[int(max_y)][int(max_x)][0]+p_cell_y[int(max_y)][int(max_x)][1])

                    frame_preds.append(ResultItem(
                        pred=Prediction(x=pred_x, y=pred_y, conf=max_conf),
                        speed={'preprocess': None, 'inference': None, 'postprocess': None }
                    ))
                    metadata.append((fid, timestamp))
            else:
                p_conf_masked = p_conf * (p_conf >= conf_threshold).float()
                max_position = torch.argmax(p_conf_masked)
                # max_y, max_x = np.unravel_index(max_position, p_conf.shape)
                max_y, max_x = np.unravel_index(max_position.cpu().numpy(), p_conf.shape)
                max_conf = p_conf[max_y, max_x].item()

                pred_x = max_x*stride + (center*stride-p_cell_x[max_y][max_x][0]+p_cell_x[max_y][max_x][1])
                pred_y = max_y*stride + (center*stride-p_cell_y[max_y][max_x][0]+p_cell_y[max_y][max_x][1])
                frame_preds.append(ResultItem(
                    pred=Prediction(x=pred_x, y=pred_y, conf=max_conf),
                    speed={'preprocess': None, 'inference': None, 'postprocess': None }
                ))
                metadata.append((fid, timestamp))
            
            result.append(ResultItem(
                pred=frame_preds if use_nms else frame_preds[0],
                speed={'preprocess': None, 'inference': None, 'postprocess': None}
            ))

            for frame_pred in frame_preds:
                pred = frame_pred.pred
                if pred.conf >= conf_threshold:
                    csv_rows.append({
                        'Frame': real_frame_idx+frame_idx,
                        'Visibility': 1,
                        'X': round(pred.x.item(), 2),
                        'Y': round(pred.y.item(), 2),
                        'Conf': round(pred.conf, 2)
                    })

        # 儲存 csv
        idx_csv = f'all.csv'
        save_csv_path = f"{csv_save_path}/{idx_csv}"
        df = pd.DataFrame(csv_rows)
        write_header = not os.path.exists(save_csv_path)
        df.to_csv(save_csv_path, mode='a', header=write_header, index=False, encoding="utf-8")
                
        if self.mqttc is not None and self.output_topic is not None:
            # Publish the results to MQTT
            self._publishPoints(frame_preds, metadata)
            LOGGER.info(f"Published {len(frame_preds)} points to MQTT topic {self.output_topic}, metadata: {metadata}, at {datetime.now()}")
        return result
    def _publishPoints(self, resultItems, metadata):
        points = []
        for i in range(len(resultItems)):
            points.append(Point(
                fid=metadata[i][0],
                timestamp=metadata[i][1],
                visibility=1,
                x=resultItems[i].pred.x,
                y=resultItems[i].pred.y,
                ))
        payload = {"linear": [p.toJson() for p in points]}
        self.mqttc.publish(self.output_topic, json.dumps(payload))
    
    # postprocess_output_file
    def postprocess_output_file(self, preds, img, orig_imgs, fids, timestamps):
        """Postprocesses predictions and returns a list of Results objects."""
        # self.profile_resources("Postprocess (before)")
        use_nms = True
        conf_threshold = 0.5
        nc = 1
        reg_max = 16
        feat_no = 8
        no = nc + reg_max * feat_no
        cell_num = 80
        stride = 8
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        proj = torch.arange(reg_max, dtype=torch.float, device=device)

        feats = preds[0].clone()
        pred_distri, pred_scores = feats.view(no, -1).split(
            (reg_max * feat_no, nc), 0)
        
        pred_scores = pred_scores.permute(1, 0).contiguous()
        pred_distri = pred_distri.permute(1, 0).contiguous()

        pred_probs = torch.sigmoid(pred_scores)
        # pred_probs = [10*self.cell_num*self.cell_num]
        
        a, c = pred_distri.shape

        pred_pos = pred_distri.view(a, feat_no, c // feat_no).softmax(2).matmul(
            proj.type(pred_distri.dtype))
        each_probs = pred_probs.view(10, cell_num, cell_num)
        each_pos_x, each_pos_y, each_pos_nx, each_pos_ny = pred_pos.view(10, cell_num, cell_num, feat_no).split([2, 2, 2, 2], dim=3)

        result = []
        for frame_idx in range(10):
            p_cell_x = each_pos_x[frame_idx]
            p_cell_y = each_pos_y[frame_idx]
            p_cell_nx = each_pos_nx[frame_idx]
            p_cell_ny = each_pos_ny[frame_idx]
            center = 0.5

            # 獲取當前圖片的 conf
            p_conf = each_probs[frame_idx]

            frame_preds = []
            if use_nms:
                nms_preds = non_max_suppression(p_conf, p_cell_x, p_cell_y, conf_threshold=conf_threshold, dis_tolerance=20)

                # 取出 nms 的結果
                for pred in nms_preds:
                    max_x, max_y, max_conf = pred
                    pred_x = max_x*stride + (center*stride-p_cell_x[int(max_y)][int(max_x)][0]+p_cell_x[int(max_y)][int(max_x)][1])
                    pred_y = max_y*stride + (center*stride-p_cell_y[int(max_y)][int(max_x)][0]+p_cell_y[int(max_y)][int(max_x)][1])

                    frame_preds.append(ResultItem(
                        pred=Prediction(x=pred_x, y=pred_y, conf=max_conf),
                        speed={'preprocess': None, 'inference': None, 'postprocess': None }
                    ))
            else:
                p_conf_masked = p_conf * (p_conf >= conf_threshold).float()
                max_position = torch.argmax(p_conf_masked)
                # max_y, max_x = np.unravel_index(max_position, p_conf.shape)
                max_y, max_x = np.unravel_index(max_position.cpu().numpy(), p_conf.shape)
                max_conf = p_conf[max_y, max_x].item()

                pred_x = max_x*stride + (center*stride-p_cell_x[max_y][max_x][0]+p_cell_x[max_y][max_x][1])
                pred_y = max_y*stride + (center*stride-p_cell_y[max_y][max_x][0]+p_cell_y[max_y][max_x][1])
                frame_preds.append(ResultItem(
                    pred=Prediction(x=pred_x, y=pred_y, conf=max_conf),
                    speed={'preprocess': None, 'inference': None, 'postprocess': None }
                ))
            
            result.append(ResultItem(
                pred=frame_preds if use_nms else frame_preds[0],
                speed={'preprocess': None, 'inference': None, 'postprocess': None}
            ))
        orig_images_clone = orig_imgs.squeeze(0).contiguous().cpu().numpy()

        p = Path(self.batch[0][0])
        parent_dir = p.parent.name
        match_dir = p.parent.parent.parent.name
        frame_save_path = os.path.join(self.save_dir, match_dir, 'frame', parent_dir)
        csv_save_path = os.path.join(self.save_dir, match_dir, 'csv', parent_dir)
        os.makedirs(frame_save_path, exist_ok=True)
        os.makedirs(csv_save_path, exist_ok=True)
        result = []
        csv_rows = []
        real_frame_idx = int(p.stem)
        for frame_idx in range(10):
            p_cell_x = each_pos_x[frame_idx]
            p_cell_y = each_pos_y[frame_idx]
            p_cell_nx = each_pos_nx[frame_idx]
            p_cell_ny = each_pos_ny[frame_idx]
            center = 0.5

            # 獲取當前圖片的 conf
            p_conf = each_probs[frame_idx]

            frame_preds = []
            if use_nms:
                nms_preds = non_max_suppression(p_conf, p_cell_x, p_cell_y, conf_threshold=conf_threshold, dis_tolerance=20)

                # 取出 nms 的結果
                for pred in nms_preds:
                    max_x, max_y, max_conf = pred
                    pred_x = max_x*stride + (center*stride-p_cell_x[int(max_y)][int(max_x)][0]+p_cell_x[int(max_y)][int(max_x)][1])
                    pred_y = max_y*stride + (center*stride-p_cell_y[int(max_y)][int(max_x)][0]+p_cell_y[int(max_y)][int(max_x)][1])

                    frame_preds.append(ResultItem(
                        pred=Prediction(x=pred_x, y=pred_y, conf=max_conf),
                        speed={'preprocess': None, 'inference': None, 'postprocess': None }
                    ))
            else:
                p_conf_masked = p_conf * (p_conf >= conf_threshold).float()
                max_position = torch.argmax(p_conf_masked)
                # max_y, max_x = np.unravel_index(max_position, p_conf.shape)
                max_y, max_x = np.unravel_index(max_position.cpu().numpy(), p_conf.shape)
                max_conf = p_conf[max_y, max_x].item()

                pred_x = max_x*stride + (center*stride-p_cell_x[max_y][max_x][0]+p_cell_x[max_y][max_x][1])
                pred_y = max_y*stride + (center*stride-p_cell_y[max_y][max_x][0]+p_cell_y[max_y][max_x][1])
                frame_preds.append(ResultItem(
                    pred=Prediction(x=pred_x, y=pred_y, conf=max_conf),
                    speed={'preprocess': None, 'inference': None, 'postprocess': None }
                ))
            
            result.append(ResultItem(
                pred=frame_preds if use_nms else frame_preds[0],
                speed={'preprocess': None, 'inference': None, 'postprocess': None}
            ))
            
            for frame_pred in frame_preds:
                pred = frame_pred.pred
                if pred.conf >= conf_threshold:
                    csv_rows.append({
                        'Frame': real_frame_idx+frame_idx,
                        'Visibility': 1,
                        'X': round(pred.x.item(), 2),
                        'Y': round(pred.y.item(), 2),
                        'Conf': round(pred.conf, 2)
                    })
            # 視覺化與儲存圖片
            img_np = orig_images_clone[frame_idx, :, :]
            img_np = img_np.astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            img_np = np.ascontiguousarray(img_np.copy())

            for frame_pred in frame_preds:
                pred = frame_pred.pred
                if pred.conf >= conf_threshold:
                    cv2.circle(img_np, (int(pred.x.item()), int(pred.y.item())), radius=3, color=(0, 0, 255), thickness=-1)
                    conf_text = f"{pred.conf:.2f}"
                    cv2.putText(img_np, conf_text, (int(pred.x.item()) + 5, int(pred.y.item()) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)
                    csv_rows.append({
                        'Frame': real_frame_idx+frame_idx,
                        'Visibility': 1,
                        'X': round(pred.x.item(), 2),
                        'Y': round(pred.y.item(), 2),
                        'Conf': round(pred.conf, 2)
                    })


            # 儲存圖片
            idx_p = f'{int(p.stem) + frame_idx}.png'
            save_img_path = f"{frame_save_path}/{idx_p}"
            self.saver.save_image(save_img_path, img_np)
        save_csv_path = os.path.join(csv_save_path, f"{p.stem}.csv")
        self.saver.save_csv(save_csv_path, csv_rows)

        
        # TODO: 這裡需要將結果轉換為原始圖片的座標系統
        # result = revert_coordinates(result, orig_imgs[0].shape[2], orig_imgs[0].shape[3], img[0].shape[2])
        # self.profile_resources("Postprocess (after)")
        return result

    def stream_inference_pro_v2(self, source=None, model=None, *args, **kwargs):
        """Optimized Asynchronous GPU Streamed Inference with torch.profiler support."""

        if not self.model:
            self.setup_model(model)
        self.setup_source(source if source is not None else self.args.source)

        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 10, *self.imgsz))
            self.done_warmup = True

        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

        num_streams = 6
        streams = [torch.cuda.Stream(priority=0) for _ in range(num_streams)]

        queue = Queue(maxsize=64)
        pending = []
        timeline_records = []

        pre_total, infer_total, post_total = 0.0, 0.0, 0.0
        total_images = 0
        feeder_finished = False

        def batch_feeder():
            nonlocal feeder_finished
            for i, batch in enumerate(dataloader):
                queue.put((i, batch))
            feeder_finished = True

        threading.Thread(target=batch_feeder, daemon=True).start()

        self.run_callbacks('on_predict_start')
        start_time = time.time()

        # ====== torch.profiler 正式啟動 ======
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=5),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_output'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            # ===================================

            any_batch_processed = False  # 用來決定是否 prof.step()

            while True:
                try:
                    while not queue.empty():
                        i, batch = queue.get_nowait()
                        self.batch = batch
                        path, im0s, vid_cap, s = batch
                        stream_idx = i % num_streams
                        stream = streams[stream_idx]

                        schedule_time = time.time() - start_time

                        # CUDA Event for each stage
                        pre_start, pre_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        infer_start, infer_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        post_start, post_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)

                        with torch.cuda.stream(stream):
                            pre_start.record()
                            im = self.preprocess(im0s)
                            pre_end.record()

                            infer_start.record()
                            preds = self.inference(im, *args, **kwargs)
                            infer_end.record()

                            post_start.record()
                            results = self.postprocess(preds, im, im0s)
                            post_end.record()

                            end_event.record()

                        pending.append({
                            "event": end_event,
                            "stream_idx": stream_idx,
                            "path": path,
                            "im0s": im0s,
                            "vid_cap": vid_cap,
                            "batch_idx": i,
                            "schedule_time": schedule_time,
                            "results": results,
                            "profiling": {
                                "pre": (pre_start, pre_end),
                                "infer": (infer_start, infer_end),
                                "post": (post_start, post_end)
                            }
                        })

                        any_batch_processed = True  # 有成功處理 batch

                except Empty:
                    pass

                next_pending = []
                for p in pending:
                    if p["event"].query():
                        complete_time = time.time() - start_time

                        timeline_records.append({
                            "batch_idx": p["batch_idx"],
                            "stream_idx": p["stream_idx"],
                            "schedule_time": p["schedule_time"],
                            "complete_time": complete_time,
                        })

                        n = p["im0s"].shape[1]
                        pre_e = p["profiling"]["pre"][0].elapsed_time(p["profiling"]["pre"][1])
                        infer_e = p["profiling"]["infer"][0].elapsed_time(p["profiling"]["infer"][1])
                        post_e = p["profiling"]["post"][0].elapsed_time(p["profiling"]["post"][1])

                        pre_total += pre_e
                        infer_total += infer_e
                        post_total += post_e
                        total_images += n

                        for j in range(n):
                            p["results"][j].speed = {
                                'preprocess': pre_e / n,
                                'inference': infer_e / n,
                                'postprocess': post_e / n
                            }
                            yield p["results"][j]

                        self.run_callbacks('on_predict_batch_end')
                        any_batch_processed = True  # 有 batch 完成
                    else:
                        next_pending.append(p)

                pending = next_pending

                if feeder_finished and queue.empty() and not pending:
                    break

                # ✅ 只有真的有 batch 被處理，才呼叫 prof.step()
                if any_batch_processed:
                    prof.step()
                    any_batch_processed = False

        self.run_callbacks('on_predict_end')

        if total_images:
            elapsed_time = time.time() - start_time
            fps = total_images / elapsed_time
            LOGGER.info(f'[SUMMARY] Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image'
                        % (pre_total / total_images, infer_total / total_images, post_total / total_images))
            LOGGER.info(f'[SUMMARY] Total elapsed time: {elapsed_time:.2f}s, Total images: {total_images}, Overall FPS: {fps:.2f}')

        self.plot_timeline(timeline_records)

    @smart_inference_mode()
    def stream_inference_profiler(self, source=None, model=None, *args, **kwargs):
        """Optimized Asynchronous GPU inference pipeline with CUDA Streams and Events."""

        if not self.model:
            self.setup_model(model)
        self.setup_source(source if source is not None else self.args.source)

        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 10, *self.imgsz))
            self.done_warmup = True

        start_time = time.time()
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=8,
        )

        profiler_output_dir = os.path.abspath("./profiler_output")
        LOGGER.info(f'profiler_output_dir: {profiler_output_dir}')

        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True
        ) as prof:

            preprocess_streams = [torch.cuda.Stream() for _ in range(15)]
            inference_streams = [torch.cuda.Stream() for _ in range(10)]
            postprocess_streams = [torch.cuda.Stream() for _ in range(15)]

            task_queue = Q.Queue(maxsize=256)
            pending = []

            pre_total, infer_total, post_total = 0.0, 0.0, 0.0
            total_images = 0
            feeder_finished = False

            def batch_feeder():
                nonlocal feeder_finished
                for i, batch in enumerate(dataloader):
                    task_queue.put((i, batch))
                feeder_finished = True

            threading.Thread(target=batch_feeder, daemon=True).start()
            self.run_callbacks('on_predict_start')

            while True:
                try:
                    while not task_queue.empty():
                        i, batch = task_queue.get_nowait()
                        path, im0s, vid_cap, s = batch

                        preprocess_stream = preprocess_streams[i % len(preprocess_streams)]
                        inference_stream = inference_streams[i % len(inference_streams)]
                        postprocess_stream = postprocess_streams[i % len(postprocess_streams)]

                        pre_start, pre_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        infer_start, infer_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        post_start, post_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)

                        # Preprocess
                        with torch.cuda.stream(preprocess_stream):
                            with torch.profiler.record_function("Preprocess"):
                                pre_start.record()
                                im = self.preprocess(im0s)
                                pre_end.record()

                        # Inference
                        with torch.cuda.stream(inference_stream):
                            inference_stream.wait_event(pre_end)
                            with torch.profiler.record_function("Inference"):
                                infer_start.record()
                                preds = self.inference(im, *args, **kwargs)
                                infer_end.record()

                        # Postprocess
                        with torch.cuda.stream(postprocess_stream):
                            postprocess_stream.wait_event(infer_end)
                            with torch.profiler.record_function("Postprocess"):
                                post_start.record()
                                results = self.postprocess(preds, im, im0s)
                                post_end.record()
                                end_event.record()

                        pending.append({
                            "event": end_event,
                            "profiling": {
                                "pre": (pre_start, pre_end),
                                "infer": (infer_start, infer_end),
                                "post": (post_start, post_end)
                            },
                            "path": path,
                            "im0s": im0s,
                            "vid_cap": vid_cap,
                            "results": results
                        })

                except Q.Empty:
                    pass

                new_pending = []
                for p in pending:
                    if p["event"].query():
                        n = p["im0s"].shape[1]
                        pre_e = p["profiling"]["pre"][0].elapsed_time(p["profiling"]["pre"][1])
                        infer_e = p["profiling"]["infer"][0].elapsed_time(p["profiling"]["infer"][1])
                        post_e = p["profiling"]["post"][0].elapsed_time(p["profiling"]["post"][1])

                        pre_total += pre_e
                        infer_total += infer_e
                        post_total += post_e
                        total_images += n

                        for j in range(n):
                            p["results"][j].speed = {
                                'preprocess': pre_e / n,
                                'inference': infer_e / n,
                                'postprocess': post_e / n
                            }

                        self.run_callbacks('on_predict_batch_end')
                        yield from p["results"]
                    else:
                        new_pending.append(p)
                pending = new_pending

                if feeder_finished and task_queue.empty() and not pending:
                    break

                prof.step()

            self.run_callbacks('on_predict_end')

        if total_images:
            elapsed_time = time.time() - start_time
            fps = total_images / elapsed_time
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape {(1, 1, *im.shape[2:])}'
                        % (pre_total / total_images, infer_total / total_images, post_total / total_images))
            LOGGER.info(f'Total elapsed time: {elapsed_time:.2f}s, Total images: {total_images}, Overall FPS: {fps:.2f}')
    @smart_inference_mode()
    def stream_inference_(self, source=None, model=None, *args, **kwargs):
        """Asynchronous GPU batch-streamed inference with maximal throughput (FPS) using CUDA Streams and Events."""

        if not self.model:
            self.setup_model(model)
        self.setup_source(source if source is not None else self.args.source)

        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 10, *self.imgsz))
            self.done_warmup = True
        start_time = time.time()
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        num_streams = 10
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        queue = Queue(maxsize=32)
        infer_queue = Queue(maxsize=32)
        pending = []

        pre_total, infer_total, post_total = 0.0, 0.0, 0.0
        total_images = 0
        feeder_finished = False

        def batch_feeder():
            nonlocal feeder_finished
            for i, batch in enumerate(dataloader):
                queue.put((i, batch))
            feeder_finished = True
        def preprocess_worker():
            while True:
                item = queue.get()
                if item is None:
                    break
                i, batch = item
                path, im0s, vid_cap, s = batch

                # CPU-side preprocess
                im = self.cpu_preprocess(im0s)  # 自己寫一個純CPU版本
                infer_queue.put((i, path, im, im0s, vid_cap, s))
                queue.task_done()
                LOGGER.info(f"[Preprocess] {i} done, queue size: {queue.qsize()}")
        

        threading.Thread(target=batch_feeder, daemon=True).start()
        threading.Thread(target=preprocess_worker, daemon=True).start()
        
        self.run_callbacks('on_predict_start')
        while True:
            while not infer_queue.empty():
                item = infer_queue.get()
                if item is None:
                    infer_queue.put(None)
                    break
                i, path, im, im0s, vid_cap, s = item
                stream = streams[i % num_streams]

                # Timing events
                pre_start, pre_end = torch.cuda.Event(True), torch.cuda.Event(True)
                infer_start, infer_end = torch.cuda.Event(True), torch.cuda.Event(True)
                post_start, post_end = torch.cuda.Event(True), torch.cuda.Event(True)
                end_event = torch.cuda.Event(True)

                with torch.cuda.stream(stream):
                    LOGGER.info(f"[Start] Stream {i % num_streams} processing batch {i} at {time.time():.4f}")
                    pre_start.record(stream)
                    im = im0s.to(self.device, dtype=torch.float32, non_blocking=True)
                    if self.model.fp16:
                        im = im.half()
                    pre_end.record(stream)

                    infer_start.record(stream)
                    preds = self.inference(im, *args, **kwargs)
                    infer_end.record(stream)

                    post_start.record(stream)
                    results = self.postprocess(preds, im, im0s)
                    post_end.record(stream)

                    end_event.record(stream)
                    LOGGER.info(f"[End] Stream {i % num_streams} finished batch {i} at {time.time():.4f}")

                pending.append({
                    "event": end_event,
                    "stream": stream,
                    "path": path,
                    "im0s": im0s,
                    "vid_cap": vid_cap,
                    "results": results,
                    "profiling": {
                        "pre": (pre_start, pre_end),
                        "infer": (infer_start, infer_end),
                        "post": (post_start, post_end)
                    }
                })

            new_pending = []
            for p in pending:
                if p["event"].query():
                    n = p["im0s"].shape[1]
                    pre_e = p["profiling"]["pre"][0].elapsed_time(p["profiling"]["pre"][1])
                    infer_e = p["profiling"]["infer"][0].elapsed_time(p["profiling"]["infer"][1])
                    post_e = p["profiling"]["post"][0].elapsed_time(p["profiling"]["post"][1])

                    pre_total += pre_e
                    infer_total += infer_e
                    post_total += post_e
                    total_images += n

                    for j in range(n):
                        p["results"][j].speed = {
                            'preprocess': pre_e / n,
                            'inference': infer_e / n,
                            'postprocess': post_e / n
                        }
                        # pj = Path(p["path"][j])
                        # im0 = None if self.source_type.tensor else p["im0s"][j].copy()

                        # if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        #     _ = self.write_results(j, p["results"], (pj, im, im0))
                        # if self.args.save or self.args.save_txt:
                        #     p["results"][j].save_dir = str(self.save_dir)
                        # if self.args.show and self.plotted_img is not None:
                        #     self.show(pj)
                        # if self.args.save and self.plotted_img is not None:
                        #     self.save_preds(p["vid_cap"], j, str(self.save_dir / pj.name))

                    self.run_callbacks('on_predict_batch_end')
                    LOGGER.info(f'{pre_e:.1f}ms {infer_e:.1f}ms {post_e:.1f}ms')
                    yield from p["results"]
                else:
                    new_pending.append(p)
            pending = new_pending

            if feeder_finished and infer_queue.empty() and not pending:
                break

        self.run_callbacks('on_predict_end')
        if total_images:
            elapsed_time = time.time() - start_time  # 單位：秒
            fps = total_images / elapsed_time
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 1, *im.shape[2:])}' % (pre_total / total_images, infer_total / total_images, post_total / total_images))
            LOGGER.info(f'Total elapsed time: {elapsed_time:.2f}s, Total images: {total_images}, Overall FPS: {fps:.2f}')


    @smart_inference_mode()
    def stream_inference_(self, source=None, model=None, *args, **kwargs):
        if not self.model:
            self.setup_model(model)
        self.setup_source(source if source is not None else self.args.source)

        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 10, *self.imgsz))
            self.done_warmup = True

        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

        # Parameters
        num_preprocess_workers = 4
        num_infer_streams = 8
        num_postprocess_workers = 2
        queue_size = 64

        preprocess_queue = Queue(maxsize=queue_size)
        infer_queue = Queue(maxsize=queue_size)
        postprocess_queue = Queue(maxsize=queue_size)

        streams = [torch.cuda.Stream() for _ in range(num_infer_streams)]
        feeder_finished = False
        start_time = time.time()

        total_images = 0
        pre_total, infer_total, post_total = 0.0, 0.0, 0.0

        # ======= Preprocess Stage =======
        def preprocess_worker():
            while True:
                item = preprocess_queue.get()
                if item is None:
                    break
                i, batch = item
                path, im0s, vid_cap, s = batch

                # CPU-side preprocess
                im = self.cpu_preprocess(im0s)  # 自己寫一個純CPU版本
                infer_queue.put((i, path, im, im0s, vid_cap, s))
                preprocess_queue.task_done()
                LOGGER.info(f"[Preprocess] {i} done, queue size: {preprocess_queue.qsize()}")

        # ======= Inference Stage =======
        def inference_worker(i, stream):
            try:
                while True:
                    item = infer_queue.get()
                    if item is None:
                        LOGGER.info(f"[inference_worker-{i}] Received shutdown signal.")
                        break

                    idx, path, im, im0s, vid_cap, s = item

                    try:
                        with torch.cuda.stream(stream):
                            # Input sanity check
                            if not torch.is_tensor(im):
                                raise ValueError(f"[inference_worker-{i}] Input is not a tensor")
                            if not torch.isfinite(im).all():
                                raise ValueError(f"[inference_worker-{i}] Input tensor contains NaN or Inf")
                            if im.dim() != 4:
                                raise ValueError(f"[inference_worker-{i}] Input tensor should be 3D (C, H, W), got {im.shape}")

                            # Send to device
                            pre_start = torch.cuda.Event(enable_timing=True)
                            infer_start = torch.cuda.Event(enable_timing=True)
                            infer_end = torch.cuda.Event(enable_timing=True)

                            pre_start.record()
                            im = im.to(self.device, dtype=torch.float32, non_blocking=True)
                            if self.model.fp16:
                                im = im.half()

                            # Inference
                            infer_start.record()
                            preds = self.inference(im, *args, **kwargs)
                            infer_end.record()

                            # Strong sync to catch CUDA error
                            torch.cuda.current_stream().synchronize()

                            # Sanity check output
                            if preds is None:
                                raise ValueError(f"[inference_worker-{i}] Inference output is None")
                            if not isinstance(preds, (list, tuple)) or len(preds) == 0:
                                raise ValueError(f"[inference_worker-{i}] Inference output invalid format: {type(preds)}")
                            
                            postprocess_queue.put((idx, path, preds, im0s, vid_cap, s, infer_start, infer_end))
                            LOGGER.info(f"[inference_worker-{i}] Finished batch {idx}")
                    except Exception as batch_e:
                        LOGGER.error(f"[inference_worker-{i}] Skipping batch {idx} due to error: {batch_e}")
                        # Don't crash entire worker, just skip this batch
                    finally:
                        infer_queue.task_done()

            except Exception as e:
                LOGGER.critical(f"[inference_worker-{i}] Fatal crash: {e}")
                raise e


        # ======= Postprocess Stage =======
        def postprocess_worker():
            nonlocal total_images, pre_total, infer_total, post_total
            pending = []

            while True:
                # 先拉新的資料
                try:
                    item = postprocess_queue.get(timeout=0.01)
                    if item is None:
                        break
                    pending.append(item)
                    postprocess_queue.task_done()
                except Exception as e:
                    LOGGER.debug(f"[postprocess_worker] queue empty: {e}")
                    pass  # queue空就算了

                # 檢查 pending 裡面有沒有完成的
                new_pending = []
                for item in pending:
                    idx, path, preds, im0s, vid_cap, s, infer_start, infer_end = item

                    if infer_end.query():  # 這個 batch 推論完成了
                        elapsed_infer = infer_start.elapsed_time(infer_end)

                        post_start_cpu = time.time()
                        results = self.cpu_postprocess(preds, im0s)
                        post_end_cpu = time.time()

                        postprocess_time = (post_end_cpu - post_start_cpu) * 1000  # ms

                        n = im0s.shape[2]
                        infer_total += elapsed_infer
                        post_total += postprocess_time
                        total_images += n

                        for j in range(n):
                            results[j].speed = {
                                'preprocess': 0.0,
                                'inference': elapsed_infer / n,
                                'postprocess': postprocess_time / n
                            }

                        self.run_callbacks('on_predict_batch_end')
                        yield from results
                    else:
                        # 還沒好，留著下一輪再檢查
                        new_pending.append(item)

                pending = new_pending


        # ======= Feeder =======
        def batch_feeder():
            nonlocal feeder_finished
            for i, batch in enumerate(dataloader):
                preprocess_queue.put((i, batch))
            feeder_finished = True

        # ======= Start Workers =======
        preprocess_workers = [threading.Thread(target=preprocess_worker, daemon=True) for _ in range(num_preprocess_workers)]
        infer_workers = [threading.Thread(target=inference_worker, args=(i, streams[i % num_infer_streams]), daemon=True) for i in range(num_infer_streams)]
        postprocess_workers = [threading.Thread(target=postprocess_worker, daemon=True) for _ in range(num_postprocess_workers)]

        for w in preprocess_workers + infer_workers + postprocess_workers:
            w.start()

        threading.Thread(target=batch_feeder, daemon=True).start()

        self.run_callbacks('on_predict_start')

        while True:
            LOGGER.info(f'[Monitor] feeder_finished={feeder_finished}, preprocess_queue={preprocess_queue.qsize()}, infer_queue={infer_queue.qsize()}, postprocess_queue={postprocess_queue.qsize()}')
            if feeder_finished and preprocess_queue.empty() and infer_queue.empty() and postprocess_queue.empty():
                break
            time.sleep(0.01)  # 避免busy loop

        # ======= Cleanup =======
        for _ in preprocess_workers:
            preprocess_queue.put(None)
        for _ in infer_workers:
            infer_queue.put(None)
        for _ in postprocess_workers:
            postprocess_queue.put(None)

        for w in preprocess_workers + infer_workers + postprocess_workers:
            w.join()

        self.run_callbacks('on_predict_end')

        torch.cuda.synchronize()

        elapsed_time = time.time() - start_time
        fps = total_images / elapsed_time
        LOGGER.info(f'Total elapsed time: {elapsed_time:.2f}s, Total images: {total_images}, Overall FPS: {fps:.2f}')
        LOGGER.info(f'Average Inference Time: {infer_total/total_images:.2f} ms')

    

    def write_results(self, idx, results, batch):
        return "todo"
    def show(self, p):
        """Display an image in a window using OpenCV imshow()."""
        return "todo"

    def save_preds(self, vid_cap, idx, save_path):
        """Save video predictions as mp4 at specified path."""
        im0 = self.plotted_img
        # Save imgs
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                suffix = '.mp4'
                fourcc = 'avc1'
                save_path = str(Path(save_path).with_suffix(suffix))
                self.vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            self.vid_writer[idx].write(im0)