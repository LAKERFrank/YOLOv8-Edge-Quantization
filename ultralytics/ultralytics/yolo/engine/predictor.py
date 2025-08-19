# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ yolo mode=predict model=yolov8n.pt source=0                               # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ yolo mode=predict model=yolov8n.pt                 # PyTorch
                              yolov8n.torchscript        # TorchScript
                              yolov8n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                              yolov8n_openvino_model     # OpenVINO
                              yolov8n.engine             # TensorRT
                              yolov8n.mlmodel            # CoreML (macOS-only)
                              yolov8n_saved_model        # TensorFlow SavedModel
                              yolov8n.pb                 # TensorFlow GraphDef
                              yolov8n.tflite             # TensorFlow Lite
                              yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov8n_paddle_model       # PaddlePaddle
"""
from datetime import datetime
import os
import platform
from pathlib import Path
import random
from threading import Thread
from queue import Empty, Queue
import threading
import time

import cv2
from matplotlib import cm, pyplot as plt
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.tracknet.utils.postprocess import PostprocessSaver
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data import load_inference_source
from ultralytics.yolo.data.augment import LetterBox, classify_transforms
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, MACOS, SETTINGS, WINDOWS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_imgsz, check_imshow
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode

STREAM_WARNING = """
    WARNING ⚠️ stream/video/webcam/dir predict source will accumulate results in RAM unless `stream=True` is passed,
    causing potential out-of-memory errors for large sources or long-running streams/videos.

    Usage:
        results = model(source=..., stream=True)  # generator of Results objects
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs
"""

class BasePredictor:
    """
    BasePredictor

    A base class for creating predictors.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_warmup (bool): Whether the predictor has finished setup.
        model (nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_path (str): Path to video file.
        vid_writer (cv2.VideoWriter): Video writer for saving video output.
        data_path (str): Path to data.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = self.get_save_dir()
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.plotted_img = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.results = None
        self.transforms = None
        # self.saver = PostprocessSaver(num_workers=self.args.workers)
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def get_save_dir(self):
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        return increment_path(Path(project) / name, exist_ok=self.args.exist_ok)

    def preprocess(self, im):
        """Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        img = im.to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        if not_tensor:
            img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def inference(self, im, *args, **kwargs):
        visualize = increment_path(self.save_dir / Path(self.batch[0][0]).stem,
                                   mkdir=True) if self.args.visualize and (not self.source_type.tensor) else False
        return self.model(im, augment=self.args.augment, visualize=visualize)

    def pre_transform(self, im):
        """Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Return: A list of transformed imgs.
        """
        same_shapes = all(x.shape == im[0].shape for x in im)
        auto = same_shapes and self.model.pt
        return [LetterBox(self.imgsz, auto=auto, stride=self.model.stride)(image=x) for x in im]

    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, _ = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.webcam or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        result = results[idx]
        log_string += result.verbose()

        if self.args.save or self.args.show:  # Add bbox to image
            plot_args = {
                'line_width': self.args.line_width,
                'boxes': self.args.boxes,
                'conf': self.args.show_conf,
                'labels': self.args.show_labels}
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            self.plotted_img = result.plot(**plot_args)
        # Write
        if self.args.save_txt:
            result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / 'crops', file_name=self.data_path.stem)

        return log_string

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions for an image and returns them."""
        return preds

    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        """Performs inference on an image or stream."""
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Result into one

    def predict_cli(self, source=None, model=None):
        """Method used for CLI prediction. It uses always generator as outputs as not required by CLI mode."""
        gen = self.stream_inference(source, model)
        for _ in gen:  # running CLI inference without accumulating any outputs (do not modify)
            pass

    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = getattr(self.model.model, 'transforms', classify_transforms(
            self.imgsz[0])) if self.args.task == 'classify' else None
        self.dataset = load_inference_source(source=source, imgsz=self.imgsz, vid_stride=self.args.vid_stride)
        self.source_type = self.dataset.source_type
        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or  # streams
                                                  len(self.dataset) > 1000 or  # images
                                                  any(getattr(self.dataset, 'video_flag', [False]))):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path, self.vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs

    @smart_inference_mode()
    def stream_inference_v1(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info('')

        # Setup model
        if not self.model:
            self.setup_model(model)

        # Setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        # Check if save_dir/ label file exists
        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

        # Warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 10, *self.imgsz))
            self.done_warmup = True
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
        self.run_callbacks('on_predict_start')
        for batch in dataloader:
            self.run_callbacks('on_predict_batch_start')
            self.batch = batch
            path, im0s, vid_cap, s, fids, timestamps = batch

            # Preprocess
            with profilers[0]:
                im = self.preprocess(im0s)

            # Inference
            with profilers[1]:
                preds = self.inference(im, *args, **kwargs)

            # Postprocess
            with profilers[2]:
                self.results = self.postprocess(preds, im, im0s, fids, timestamps)
            self.run_callbacks('on_predict_postprocess_end')

            # Visualize, save, write results
            n = im0s.shape[1]
            for i in range(n):
                self.seen += 1
                self.results[i].speed = {
                    'preprocess': profilers[0].dt * 1E3 / n,
                    'inference': profilers[1].dt * 1E3 / n,
                    'postprocess': profilers[2].dt * 1E3 / n}
                # p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                # p = Path(p)

                # if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                #     s += self.write_results(i, self.results, (p, im, im0))
                # if self.args.save or self.args.save_txt:
                #     self.results[i].save_dir = self.save_dir.__str__()
                # if self.args.show and self.plotted_img is not None:
                #     self.show(p)
                # if self.args.save and self.plotted_img is not None:
                #     self.save_preds(vid_cap, i, str(self.save_dir / p.name))
            LOGGER.info(f'{profilers[0].dt * 1E3:.1f}ms {profilers[1].dt * 1E3:.1f}ms {profilers[2].dt * 1E3:.1f}ms')
            self.run_callbacks('on_predict_batch_end')
            yield from self.results

            # Print time (inference-only)
            if self.args.verbose:
                # LOGGER.info(f'{s}{profilers[1].dt * 1E3:.1f}ms')
                LOGGER.info(f'{profilers[0].dt * 1E3:.1f}ms {profilers[1].dt * 1E3:.1f}ms')

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer

        self.saver.flush()  # flush all postprocess tasks

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in profilers)  # speeds per image
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 1, *im.shape[2:])}' % t)
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks('on_predict_end')

    # stream_inference_profiler
    @smart_inference_mode()
    def stream_inference_profiler(self, source=None, model=None, *args, **kwargs):
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
            prefetch_factor=8,
        )
        num_streams = 10
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        queue = Queue(maxsize=256)
        pending = []

        pre_total, infer_total, post_total = 0.0, 0.0, 0.0
        total_images = 0
        feeder_finished = False

        def batch_feeder():
            nonlocal feeder_finished
            for i, batch in enumerate(dataloader):
                queue.put((i, batch))
            feeder_finished = True

        Thread(target=batch_feeder, daemon=True).start()
        self.run_callbacks('on_predict_start')

        profiler_output_dir = os.path.abspath("./profiler_output")
        os.makedirs(profiler_output_dir, exist_ok=True)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_output_dir),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
        ) as prof:

            while True:
                try:
                    while not queue.empty():
                        i, batch = queue.get_nowait()
                        self.batch = batch
                        path, im0s, vid_cap, s = batch
                        stream = streams[i % num_streams]

                        # Timing events
                        pre_start, pre_end = torch.cuda.Event(True), torch.cuda.Event(True)
                        infer_start, infer_end = torch.cuda.Event(True), torch.cuda.Event(True)
                        post_start, post_end = torch.cuda.Event(True), torch.cuda.Event(True)
                        end_event = torch.cuda.Event(True)

                        with torch.cuda.stream(stream):
                            with torch.profiler.record_function("Preprocess"):
                                pre_start.record(stream)
                                im = self.preprocess(im0s)
                                pre_end.record(stream)
                            with torch.profiler.record_function("Inference"):
                                infer_start.record(stream)
                                preds = self.inference(im, *args, **kwargs)
                                infer_end.record(stream)
                            with torch.profiler.record_function("Postprocess"):
                                post_start.record(stream)
                                results = self.postprocess(preds, im, im0s)
                                post_end.record(stream)

                            end_event.record(stream)

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
                        prof.step()

                except Queue.Empty:
                    LOGGER.debug(f"Queue is empty, waiting for new batches...")
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

                if feeder_finished and queue.empty() and not pending:
                    break

        self.run_callbacks('on_predict_end')
        if total_images:
            elapsed_time = time.time() - start_time
            fps = total_images / elapsed_time
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 1, *im.shape[2:])}' % (pre_total / total_images, infer_total / total_images, post_total / total_images))
            LOGGER.info(f'Total elapsed time: {elapsed_time:.2f}s, Total images: {total_images}, Overall FPS: {fps:.2f}')


    # stream_inference_single_stream
    @smart_inference_mode()
    def stream_inference_single_stream(self, source=None, model=None, *args, **kwargs):
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
            prefetch_factor=8,
        )
        num_streams = 10
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        queue = Queue(maxsize=256)
        pending = []

        pre_total, infer_total, post_total = 0.0, 0.0, 0.0
        total_images = 0
        feeder_finished = False

        def batch_feeder():
            nonlocal feeder_finished
            for i, batch in enumerate(dataloader):
                queue.put((i, batch))
            feeder_finished = True

        Thread(target=batch_feeder, daemon=True).start()
        self.run_callbacks('on_predict_start')

        while True:
            try:
                while not queue.empty():
                    i, batch = queue.get_nowait()
                    self.batch = batch
                    path, im0s, vid_cap, s = batch
                    stream = streams[i % num_streams]

                    # Timing events
                    pre_start, pre_end = torch.cuda.Event(True), torch.cuda.Event(True)
                    infer_start, infer_end = torch.cuda.Event(True), torch.cuda.Event(True)
                    post_start, post_end = torch.cuda.Event(True), torch.cuda.Event(True)
                    end_event = torch.cuda.Event(True)

                    with torch.cuda.stream(stream):
                        LOGGER.info(f"[Start] Stream {i % num_streams} processing batch {i} at {time.time():.4f}")
                        pre_start.record(stream)
                        im = self.preprocess(im0s)
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
            except Queue.Empty:
                LOGGER.debug(f"Queue is empty, waiting for new batches...")
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
                    # LOGGER.info(f'{pre_e:.1f}ms {infer_e:.1f}ms {post_e:.1f}ms')
                    yield from p["results"]
                else:
                    new_pending.append(p)
            pending = new_pending

            if feeder_finished and queue.empty() and not pending:
                break

        self.run_callbacks('on_predict_end')
        if total_images:
            elapsed_time = time.time() - start_time  # 單位：秒
            fps = total_images / elapsed_time
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 1, *im.shape[2:])}' % (pre_total / total_images, infer_total / total_images, post_total / total_images))
            LOGGER.info(f'Total elapsed time: {elapsed_time:.2f}s, Total images: {total_images}, Overall FPS: {fps:.2f}')

    # stream_inference_multiple_stream
    @smart_inference_mode()
    def stream_inference_multiple_stream(self, source=None, model=None, *args, **kwargs):
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
            num_workers=16,
            pin_memory=True,
            prefetch_factor=16,
        )
        preprocess_num_streams = 2
        inference_num_streams = 1
        postprocess_num_streams = 1

        preprocess_streams = [torch.cuda.Stream() for _ in range(preprocess_num_streams)]
        inference_streams = [torch.cuda.Stream() for _ in range(inference_num_streams)]
        postprocess_streams = [torch.cuda.Stream() for _ in range(postprocess_num_streams)]

        queue = Queue(maxsize=1000)
        pending = []

        pre_total, infer_total, post_total = 0.0, 0.0, 0.0
        total_images = 0
        feeder_finished = False

        def batch_feeder():
            nonlocal feeder_finished
            for i, batch in enumerate(dataloader):
                queue.put((i, batch))
            feeder_finished = True

        Thread(target=batch_feeder, daemon=True).start()
        self.run_callbacks('on_predict_start')

        while True:
            while not queue.empty():
                i, batch = queue.get_nowait()
                self.batch = batch
                path, im0s, vid_cap, s = batch

                preprocess_stream = preprocess_streams[i % preprocess_num_streams]
                inference_stream = inference_streams[i % inference_num_streams]
                postprocess_stream = postprocess_streams[i % postprocess_num_streams]

                # CUDA Events
                pre_start, pre_end = torch.cuda.Event(True), torch.cuda.Event(True)
                infer_start, infer_end = torch.cuda.Event(True), torch.cuda.Event(True)
                post_start, post_end = torch.cuda.Event(True), torch.cuda.Event(True)
                end_event = torch.cuda.Event(True)

                # Step 1: Preprocess
                with torch.cuda.stream(preprocess_stream):
                    pre_start.record(preprocess_stream)
                    im = self.preprocess(im0s)
                    pre_end.record(preprocess_stream)

                # Step 2: Inference (必須等待 preprocess 完成)
                with torch.cuda.stream(inference_stream):
                    inference_stream.wait_event(pre_end)
                    infer_start.record(inference_stream)
                    preds = self.inference(im, *args, **kwargs)
                    infer_end.record(inference_stream)

                # Step 3: Postprocess (必須等待 inference 完成)
                with torch.cuda.stream(postprocess_stream):
                    postprocess_stream.wait_event(infer_end)
                    post_start.record(postprocess_stream)
                    results = self.postprocess(preds, im, im0s)
                    post_end.record(postprocess_stream)

                    end_event.record(postprocess_stream)

                pending.append({
                    "event": end_event,
                    "stream": postprocess_stream,
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
                    path = p["path"]
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
                    # LOGGER.info(f'{path}: {pre_e:.1f}ms {infer_e:.1f}ms {post_e:.1f}ms')
                    yield from p["results"]
                else:
                    new_pending.append(p)
            pending = new_pending

            if feeder_finished and queue.empty() and not pending:
                break

        self.run_callbacks('on_predict_end')
        if total_images:
            elapsed_time = time.time() - start_time
            fps = total_images / elapsed_time
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 1, *im.shape[2:])}' % (pre_total / total_images, infer_total / total_images, post_total / total_images))
            LOGGER.info(f'Total elapsed time: {elapsed_time:.2f}s, Total images: {total_images}, Overall FPS: {fps:.2f}')

    # stream_inference_single_stream_v2 1010FPS
    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Optimized Asynchronous GPU Streamed Inference with timeline recording and visualization."""

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
            num_workers=0,
            pin_memory=True,
            prefetch_factor=None,
            persistent_workers=False,
        )

        num_streams = 6
        streams = [torch.cuda.Stream(priority=0) for _ in range(num_streams)]

        max_pending_batches = min(64, torch.cuda.get_device_properties(0).multi_processor_count * 2)
        queue = Queue(maxsize=max_pending_batches)
        print(f"Max pending batches: {max_pending_batches}")

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
        
        while True:
            try:
                stream_count = 0
                while not queue.empty() and stream_count < max_pending_batches:
                    i, batch = queue.get_nowait()
                    self.batch = batch
                    path, im0s, vid_cap, s, fids, timestamps = batch
                    stream_idx = i % num_streams
                    stream = streams[stream_idx]

                    schedule_time = time.time() - start_time

                    # LOGGER.info(f"[SCHEDULER] Assign batch {i} to Stream-{stream_idx} at {schedule_time:.6f}s")

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
                        results = self.postprocess(preds, im, im0s, fids, timestamps)
                        post_end.record()

                        end_event.record()
                        # LOGGER.info(f'batch{i} scheduled on {time.time():.6f}')
                        

                    pending.append({
                        "event": end_event,
                        "stream_idx": stream_idx,
                        "path": path,
                        "im0s": im0s,
                        "vid_cap": vid_cap,
                        "batch_idx": i,
                        "schedule_time": schedule_time,   # <<< 記錄
                        "results": results,
                        "profiling": {
                            "pre": (pre_start, pre_end),
                            "infer": (infer_start, infer_end),
                            "post": (post_start, post_end)
                        }
                    })
                    stream_count += 1

            except Empty:
                pass
            
            next_pending = []
            for p in pending:
                if p["event"].query():
                    complete_time = time.time() - start_time

                    # 記錄 timeline
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

                    LOGGER.info(f"[COMPLETE] Batch {p['batch_idx']} on Stream-{p['stream_idx']} "
                                f"Pre: {pre_e:.2f}ms, Infer: {infer_e:.2f}ms, Post: {post_e:.2f}ms")

                    for j in range(n):
                        p["results"][j].speed = {
                            'preprocess': pre_e / n,
                            'inference': infer_e / n,
                            'postprocess': post_e / n
                        }
                        yield p["results"][j]

                    self.run_callbacks('on_predict_batch_end')
                else:
                    LOGGER.info(f"[PENDING] Batch {p['batch_idx']} on Stream-{p['stream_idx']} is still pending")
                    next_pending.append(p)

            pending = next_pending

            if feeder_finished and queue.empty() and not pending:
                break

        self.run_callbacks('on_predict_end')

        if total_images:
            elapsed_time = time.time() - start_time
            fps = total_images / elapsed_time
            LOGGER.info(f'[SUMMARY] Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image'
                        % (pre_total / total_images, infer_total / total_images, post_total / total_images))
            LOGGER.info(f'[SUMMARY] Total elapsed time: {elapsed_time:.2f}s, Total images: {total_images}, Overall FPS: {fps:.2f}')

        # 畫 timeline
        # self.plot_timeline(timeline_records)

    @smart_inference_mode()
    def stream_inference_single_stream_v2_with_gpu_plot(self, source=None, model=None, *args, **kwargs):
        """Optimized Asynchronous GPU Streamed Inference with Timeline and GPU/CPU Memory Monitoring."""

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

        process = psutil.Process(os.getpid())  # 取得當前程序

        def batch_feeder():
            nonlocal feeder_finished
            for i, batch in enumerate(dataloader):
                queue.put((i, batch))
            feeder_finished = True

        threading.Thread(target=batch_feeder, daemon=True).start()

        self.run_callbacks('on_predict_start')
        start_time = time.time()

        while True:
            try:
                while not queue.empty():
                    i, batch = queue.get_nowait()
                    self.batch = batch
                    path, im0s, vid_cap, s = batch
                    stream_idx = i % num_streams
                    stream = streams[stream_idx]

                    schedule_time = time.time() - start_time

                    # 記錄memory
                    gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    cpu_mem = process.memory_info().rss / 1024 / 1024      # MB

                    LOGGER.info(f"[SCHEDULER] Assign batch {i} to Stream-{stream_idx} at {schedule_time:.6f}s")

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
                        "gpu_mem": gpu_mem,
                        "cpu_mem": cpu_mem,
                        "profiling": {
                            "pre": (pre_start, pre_end),
                            "infer": (infer_start, infer_end),
                            "post": (post_start, post_end)
                        }
                    })

            except Empty:
                pass

            next_pending = []
            for p in pending:
                if p["event"].query():
                    complete_time = time.time() - start_time

                    # 完成後再記錄memory
                    gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
                    cpu_mem = process.memory_info().rss / 1024 / 1024

                    timeline_records.append({
                        "batch_idx": p["batch_idx"],
                        "stream_idx": p["stream_idx"],
                        "schedule_time": p["schedule_time"],
                        "complete_time": complete_time,
                        "gpu_mem": gpu_mem,
                        "cpu_mem": cpu_mem,
                    })

                    n = p["im0s"].shape[1]
                    pre_e = p["profiling"]["pre"][0].elapsed_time(p["profiling"]["pre"][1])
                    infer_e = p["profiling"]["infer"][0].elapsed_time(p["profiling"]["infer"][1])
                    post_e = p["profiling"]["post"][0].elapsed_time(p["profiling"]["post"][1])

                    pre_total += pre_e
                    infer_total += infer_e
                    post_total += post_e
                    total_images += n

                    LOGGER.info(f"[COMPLETE] Batch {p['batch_idx']} on Stream-{p['stream_idx']} "
                                f"Pre: {pre_e:.2f}ms, Infer: {infer_e:.2f}ms, Post: {post_e:.2f}ms, "
                                f"Finished at {complete_time:.6f}s")

                    for j in range(n):
                        p["results"][j].speed = {
                            'preprocess': pre_e / n,
                            'inference': infer_e / n,
                            'postprocess': post_e / n
                        }
                        yield p["results"][j]

                    self.run_callbacks('on_predict_batch_end')
                else:
                    next_pending.append(p)

            pending = next_pending

            if feeder_finished and queue.empty() and not pending:
                break

        self.run_callbacks('on_predict_end')

        if total_images:
            elapsed_time = time.time() - start_time
            fps = total_images / elapsed_time
            LOGGER.info(f'[SUMMARY] Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image'
                        % (pre_total / total_images, infer_total / total_images, post_total / total_images))
            LOGGER.info(f'[SUMMARY] Total elapsed time: {elapsed_time:.2f}s, Total images: {total_images}, Overall FPS: {fps:.2f}')

        self.plot_timeline_gpu(timeline_records)
    def plot_timeline_gpu(self, timeline_records):
        output_dir = './profiler_output'
        os.makedirs(output_dir, exist_ok=True)

        fig, ax1 = plt.subplots(figsize=(18, 10))
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

        # Plot batches as bars
        for record in timeline_records:
            stream = record['stream_idx']
            batch = record['batch_idx']
            start = record['schedule_time']
            end = record['complete_time']
            ax1.barh(
                y=f"Stream-{stream}",
                width=end - start,
                left=start,
                height=0.4,
                color=colors[stream % len(colors)],
                edgecolor='black'
            )
            ax1.text(start + (end - start) / 2, f"Stream-{stream}", f"B{batch}", ha='center', va='center', fontsize=6)

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Streams')
        ax1.set_title('Inference Timeline + Memory Usage')
        ax1.grid(True)

        # Plot Memory Usage
        ax2 = ax1.twinx()
        times = [r['complete_time'] for r in timeline_records]
        gpu_mems = [r['gpu_mem'] for r in timeline_records]
        cpu_mems = [r['cpu_mem'] for r in timeline_records]

        ax2.plot(times, gpu_mems, label='GPU Memory (MB)', color='cyan', linewidth=2)
        ax2.plot(times, cpu_mems, label='CPU Memory (MB)', color='magenta', linewidth=2, linestyle='--')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.legend(loc='upper right')

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f"timeline_memory_{timestamp}.png")
        plt.savefig(save_path)
        plt.close(fig)

        LOGGER.info(f"[Profiler] Timeline with memory usage saved to {save_path}")
    def plot_timeline(self, timeline_records):
        import matplotlib.pyplot as plt
        output_dir = './profiler_output'
        os.makedirs(output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(16, 8))

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

        for record in timeline_records:
            stream = record['stream_idx']
            batch = record['batch_idx']
            start = record['schedule_time']
            end = record['complete_time']                 # LOGGER.info(f"[COMPLETE] Batch {p['batch_idx']} on Stream-{p['stream_idx']} "    
            ax.barh(
                y=f"Stream-{stream}",
                width=end - start,
                left=start,
                height=0.4,
                color=colors[stream % len(colors)],
                edgecolor='black'
            )
            ax.text(start + (end - start) / 2, stream, f"B{batch}", ha='center', va='center', fontsize=8)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Streams')
        ax.set_title('Inference Timeline')
        plt.grid(True)
        plt.tight_layout()
        # 自動生成 filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f"timeline_{timestamp}.png")

        plt.savefig(save_path)
        plt.close(fig)  # 重要！釋放記憶體
        print(f"[Profiler] Timeline saved to {save_path}")


    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        self.model = AutoBackend(model or self.args.model,
                                 device=select_device(self.args.device, verbose=verbose),
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)

        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        self.model.eval()

    def show(self, p):
        """Display an image in a window using OpenCV imshow()."""
        im0 = self.plotted_img
        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(500 if self.batch[3].startswith('image') else 1)  # 1 millisecond

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
                suffix = '.mp4' if MACOS else '.avi' if WINDOWS else '.avi'
                fourcc = 'avc1' if MACOS else 'WMV2' if WINDOWS else 'MJPG'
                save_path = str(Path(save_path).with_suffix(suffix))
                self.vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            self.vid_writer[idx].write(im0)

    def run_callbacks(self, event: str):
        """Runs all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func):
        """
        Add callback
        """
        self.callbacks[event].append(func)
