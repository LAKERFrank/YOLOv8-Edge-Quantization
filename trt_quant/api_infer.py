"""Unified inference API for TensorRT pose models."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .postprocess import postprocess_batch
from .preprocess import preprocess_3
from .trt_runner import TrtRunner


class PoseTRTInfer:
    """TensorRT pose inference wrapper matching PyTorch output format."""

    def __init__(
        self,
        engine_path: str,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        to_gray: bool = False,
        letterbox: bool = True,
        profile_index: int = 0,
        device_index: int = 0,
        nc: int = 1,
        nkpt: int = 17,
        names: dict[int, str] | None = None,
    ) -> None:
        self.runner = TrtRunner(engine_path, profile_index=profile_index, device_index=device_index)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.to_gray = to_gray
        self.letterbox = letterbox
        self.nc = nc
        self.nkpt = nkpt
        self.names = names

    def infer_3in3out(self, frames3: Sequence[np.ndarray], verbose: bool = False) -> List:
        x, metas = preprocess_3(frames3, self.imgsz, to_gray=self.to_gray, letterbox=self.letterbox)
        outputs = self.runner.infer(x, verbose=verbose)
        results = postprocess_batch(
            outputs,
            metas,
            conf=self.conf,
            iou=self.iou,
            nc=self.nc,
            nkpt=self.nkpt,
            names=self.names,
            verbose=verbose,
        )
        return results
