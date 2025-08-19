import os
import time
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from glob import glob
from tqdm import tqdm

from ultralytics.yolo.data.build import check_source
from ultralytics.yolo.data.dataloaders.stream_loaders import SourceTypes

class TrackNetPredDataset(Dataset):
    def __init__(self, dir, num_input=10, imgsz=640, transform=None, prefix=''):
        self.dir = dir
        self.transform = transform
        self.num_input = num_input
        self.imgsz = imgsz
        self.prefix = prefix
        self.samples = []
        self.bs = 1
        source, webcam, screenshot, from_img, in_memory, tensor = check_source(self.dir)
        self.source_type = source.source_type if in_memory else SourceTypes(webcam, screenshot, from_img, tensor)

        img_files = sorted(glob(os.path.join(dir, prefix + "*.png")),
                           key=lambda x: int(os.path.basename(x).split('.')[0]))

        total_batches = len(img_files) // num_input
        fid = 0
        for i in tqdm(range(total_batches), desc="Loading batches", ncols=80):
            img_files_10 = img_files[i*self.num_input : i*self.num_input + self.num_input]

            frames = []
            fids = []
            timestamps = []
            for fp in img_files_10:
                img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(f"Cannot load image {fp}")
                img = img.astype(np.float32)
                img = self.pad_to_square(img)
                img = cv2.resize(img, dsize=(self.imgsz, self.imgsz), interpolation=cv2.INTER_CUBIC)
                img = np.expand_dims(img, axis=0)  # (1, H, W)
                frames.append(img)
                fids.append(fid)
                fid += 1
                timestamps.append(time.time())

            img = np.concatenate(frames, 0)  # (num_input, H, W)

            if self.transform:
                img = self.transform(img)

            img_tensor = torch.from_numpy(img).float()  # (num_input, H, W)
            self.samples.append((img_files_10[0], img_tensor, "", "", fids, timestamps))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]  # 直接 memory lookup

    def pad_to_square(self, img, pad_value=0):
        h, w = img.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = (0, 0, pad1, pad2) if h > w else (pad1, pad2, 0, 0)
        img = cv2.copyMakeBorder(img, *pad, borderType=cv2.BORDER_CONSTANT, value=pad_value)
        return img
