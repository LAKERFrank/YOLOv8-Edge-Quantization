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
        to_gray: bool | None = None,
    ) -> None:
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.runner = TrtRunner(engine_path, profile_index=profile_index)
        if to_gray is None:
            channel_count = self.runner.get_input_channel_count()
            self.to_gray = channel_count == 1
        else:
            self.to_gray = to_gray

    def infer_3in3out(self, frames3: Sequence[np.ndarray]) -> List[Result]:
        x, metas = preprocess_3(frames3, self.imgsz, to_gray=self.to_gray)
        outputs = self.runner.infer(x, verbose=True)
        results = postprocess_batch(outputs, metas, self.conf, self.iou, verbose=True)
        return results
