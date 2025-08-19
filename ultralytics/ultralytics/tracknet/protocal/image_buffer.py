import time
from typing import Protocol
import numpy as np
import threading

class FrameProtocol(Protocol):
    is_eos: bool

    @property
    def height(self) -> int: ...
    @property
    def image(self) -> np.ndarray[np.uint8]: ...
    @property
    def index(self) -> int: ...
    @property
    def monotonic_timestamp(self) -> float: ...
    @property
    def timestamp(self) -> float: ...
    @property
    def width(self) -> int: ...

class ImageBufferProtocol(Protocol):
    def clear(self) -> None: ...
    def pop(self, blocking: bool = True) -> FrameProtocol: ...
    def push(self, frame: FrameProtocol) -> None: ...

class FakeFrame:
    def __init__(self, image: np.ndarray, index: int, is_eos=False):
        self._image = image
        self._index = index
        self._is_eos = is_eos
        self._timestamp = time.time()
        self._monotonic_timestamp = time.monotonic()

    @property
    def image(self) -> np.ndarray:
        return self._image

    @property
    def index(self) -> int:
        return self._index

    @property
    def is_eos(self) -> bool:
        return self._is_eos

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def monotonic_timestamp(self) -> float:
        return self._monotonic_timestamp

    @property
    def height(self) -> int:
        return self._image.shape[0]

    @property
    def width(self) -> int:
        return self._image.shape[1]

class FakeImageBuffer:
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)

    def push(self, frame):
        with self.cv:
            self.queue.append(frame)
            self.cv.notify()

    def pop(self, blocking=True):
        with self.cv:
            while blocking and not self.queue:
                self.cv.wait()
            return self.queue.pop(0)

    def clear(self):
        with self.cv:
            self.queue.clear()