import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ultralytics.tracknet.protocal.image_buffer import ImageBufferProtocol

class ImageBufferDataset(Dataset):
    def __init__(self, image_buffer: ImageBufferProtocol, track_size: int = 10):
        self.image_buffer = image_buffer
        self.track_size = track_size
        self.buffer = []  # 儲存一組 track_size 連續 frame
        self.imgsz = 640

    def __len__(self):
        return 1_000_000  # 任意大，會由外部控制終止（如遇到 EOS）
    
    def pad_to_square(self, img, pad_value=0):
        h, w = img.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = (0, 0, pad1, pad2) if h > w else (pad1, pad2, 0, 0)
        img = cv2.copyMakeBorder(img, *pad, borderType=cv2.BORDER_CONSTANT, value=pad_value)
        return img
    
    def __getitem__(self, index):
        frames = []
        fids = []
        timestamps = []

        while len(frames) < self.track_size:
            frame = self.image_buffer.pop(True)
            if frame.is_eos:
                raise StopIteration  # 結束迴圈
            img = frame.image.astype(np.float32)
            img = self.pad_to_square(img)
            img = cv2.resize(img, dsize=(self.imgsz, self.imgsz), interpolation=cv2.INTER_CUBIC)
            img = np.expand_dims(img, axis=0)  # (1, H, W)
            frames.append(img)
            fids.append(frame.index)
            timestamps.append(frame.monotonic_timestamp)

        img = np.concatenate(frames, 0)
        img_tensor = torch.from_numpy(img).float()
            
        return (f"stream_dataset_{index}", img_tensor, "", "", fids, timestamps)