from typing import List, Sequence, Tuple

import numpy as np

from preprocess import preprocess_3
from postprocess import Result, postprocess_batch
from trt_runner import TrtRunner


class PoseTRTInfer:
    def __init__(
        self,
        engine_path: str,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: Tuple[int, int] = (640, 640),
        profile_index: int = 0,
    ) -> None:
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.runner = TrtRunner(engine_path, profile_index=profile_index)

    def infer_3in3out(self, frames3: Sequence[np.ndarray]) -> List[Result]:
        x, metas = preprocess_3(frames3, self.imgsz)
        outputs = self.runner.infer(x, verbose=True)
        results = postprocess_batch(outputs, metas, self.conf, self.iou, verbose=True)
        return results
