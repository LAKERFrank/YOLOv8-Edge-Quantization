from matplotlib import patches, pyplot as plt
import torch
from torch.utils.data import Dataset
import cv2
import hashlib
import os
import pandas as pd
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from functools import lru_cache
from glob import glob

class TrackNetDataset(Dataset):
    def __init__(self, root_dir, num_input=10, transform=None, prefix=''):

        self.root_dir = root_dir
        self.transform = transform
        self.num_input = num_input
        self.samples = []
        self.prefix = prefix

        self.idx = set()

        image_count = len(glob(os.path.join(self.root_dir, "*/", "frame/", "*/", "*.png")))

        self.pbar = tqdm(total=image_count+(image_count-self.num_input*2+1)
                         +(image_count-self.num_input*3+1), miniters=1, smoothing=1)
        # Traverse all matches
        for match_name in glob("*/", root_dir=root_dir):
            match_name = match_name.strip('/')

            match_dir_path = os.path.join(root_dir, match_name)
            
            # Check if it is a match directory
            if not os.path.isdir(match_dir_path):
                continue

            self.read_match(match_name)
        self.pbar.close()

    def read_match(self, match_name):
        video_dir = os.path.join(self.root_dir, match_name, 'video')
        csv_dir = os.path.join(self.root_dir, match_name, 'csv')

        self.pbar.set_description(f'{self.prefix} Generating image cache: {match_name}/ ')

        # Traverse all videos in the match directory
        for video_name in glob("*.mp4", root_dir=video_dir):
            video_name = video_name.removesuffix('.mp4')

            csv_file = os.path.join(csv_dir, video_name + "_ball" + '.csv')
            
            ball_trajectory_df = self.__preprocess_csv(csv_file)

            #print(ball_trajectory_df.columns)
            #['Frame', 'Visibility', 'X', 'Y', 'dX', 'dY', 'hit']

            frame_dir = os.path.join(self.root_dir, match_name, 'frame', video_name)

            img_files = sorted(glob("*.png", root_dir=frame_dir), key=lambda x: int(x.removesuffix(".png")))

            # open first image with out coordinates
            img = cv2.cvtColor(cv2.imread(os.path.join(frame_dir, img_files[0])), cv2.COLOR_BGR2GRAY)
            h, w = img.shape

            # Create sliding windows of num_input frames
            for i in range(len(img_files) - (self.num_input-1)):
                self.pbar.update(1)

                frames = img_files[i: i + self.num_input]

                target = ball_trajectory_df.iloc[i: i + self.num_input].values
                target = self.transform_coordinates(target, w, h)

                # Avoid invalid data
                if len(frames) == self.num_input and len(target) == self.num_input:
                    npy_path = self.img_cache_dir(match_name, video_name, frames)

                    self.samples.append({
                        "match_name": match_name,
                        "video_name": video_name,
                        "cache_npy": npy_path,
                        "img_files": frames,
                        "target": target
                    })

                    self.img_cache(match_name, video_name, frames, npy_path)

                    hit_exists = np.any(target[:, 6] == 1)
                    if hit_exists:
                        for i in range(5):
                            self.samples.append({
                                "match_name": match_name,
                                "video_name": video_name,
                                "cache_npy": npy_path,
                                "img_files": frames,
                                "target": target
                            })
            
            # 降低 FPS 120 => 60
            for i in range(len(img_files) - (self.num_input*2-1)):
                self.pbar.update(1)

                frames = img_files[i: i + self.num_input*2: 2]

                target = ball_trajectory_df.iloc[i: i + self.num_input*2: 2].values
                target = self.transform_coordinates(target, w, h)

                # Avoid invalid data
                if len(frames) == self.num_input and len(target) == self.num_input:
                    npy_path = self.img_cache_dir(match_name, video_name, frames)

                    self.samples.append({
                        "match_name": match_name,
                        "video_name": video_name,
                        "cache_npy": npy_path,
                        "img_files": frames,
                        "target": target
                    })

                    self.img_cache(match_name, video_name, frames, npy_path)

                    hit_exists = np.any(target[:, 6] == 1)
                    if hit_exists:
                        for i in range(5):
                            self.samples.append({
                                "match_name": match_name,
                                "video_name": video_name,
                                "cache_npy": npy_path,
                                "img_files": frames,
                                "target": target
                            })
            # # 降低 FPS 120 => 40
            # for i in range(len(img_files) - (self.num_input*3-1)):
            #     self.pbar.update(1)

            #     frames = img_files[i: i + self.num_input*3: 3]

            #     target = ball_trajectory_df.iloc[i: i + self.num_input*3: 3].values
            #     target = self.transform_coordinates(target, w, h)

            #     # Avoid invalid data
            #     if len(frames) == self.num_input and len(target) == self.num_input:
            #         npy_path = self.img_cache_dir(match_name, video_name, frames)

            #         self.samples.append({
            #             "match_name": match_name,
            #             "video_name": video_name,
            #             "cache_npy": npy_path,
            #             "img_files": frames,
            #             "target": target
            #         })

            #         self.img_cache(match_name, video_name, frames, npy_path)

            #         hit_exists = np.any(target[:, 6] == 1)
            #         if hit_exists:
            #             for i in range(5):
            #                 self.samples.append({
            #                     "match_name": match_name,
            #                     "video_name": video_name,
            #                     "cache_npy": npy_path,
            #                     "img_files": frames,
            #                     "target": target
            #                 })
            # 降低 FPS 120 => 30
            # for i in range(len(img_files) - (self.num_input*4-1)):
            #     self.pbar.update(1)

            #     frames = img_files[i: i + self.num_input*4: 4]

            #     target = ball_trajectory_df.iloc[i: i + self.num_input*4: 4].values
            #     target = self.transform_coordinates(target, w, h)

            #     # Avoid invalid data
            #     if len(frames) == self.num_input and len(target) == self.num_input:
            #         npy_path = self.img_cache_dir(match_name, video_name, frames)

            #         self.samples.append({
            #             "match_name": match_name,
            #             "video_name": video_name,
            #             "cache_npy": npy_path,
            #             "img_files": frames,
            #             "target": target
            #         })

            #         self.img_cache(match_name, video_name, frames, npy_path)

            self.pbar.update(self.num_input-1)

    def img_cache_dir(self, match_name, video_name, img_files):
        s = '|'.join([match_name]+[video_name]+img_files)
        filename = hashlib.sha1(s.encode('utf-8')).hexdigest()

        if filename in self.idx:
            raise Exception('DUP: '+filename)
        self.idx.add(filename)

        d = os.path.join(self.root_dir, ".cache", filename[:2], filename[2:4])
        os.makedirs(d, exist_ok=True)
        f = os.path.join(d, f"{filename}.npy")
        return f

    def img_cache(self, match_name, video_name, img_files, npy_path):

        if os.path.isfile(npy_path):
            return

        # generate cache
        images = [self.__preprocess_img(os.path.join(self.root_dir, match_name, 'frame', video_name, img_file)) for img_file in img_files]
        img = np.concatenate(images, 0)

        np.save(npy_path, img)

    def get_image_cache(self, path):
        try:
            return np.load(path)
        except Exception as e:
            raise Exception("File corrupted: " + path)

    def __preprocess_csv(self, csv_file):
        # Read the ball_trajectory csv file
        ball_trajectory_df = pd.read_csv(csv_file)
        ball_trajectory_df['dX'] = -1*ball_trajectory_df['X'].diff(-1).fillna(0)
        ball_trajectory_df['dY'] = -1*ball_trajectory_df['Y'].diff(-1).fillna(0)

        if 'Event' in ball_trajectory_df.columns:
            ball_trajectory_df['hit'] = ((ball_trajectory_df['Event'] == 1) | (ball_trajectory_df['Event'] == 2)).astype(int)
        else:
            ball_trajectory_df['hit'] = 0

        #ball_trajectory_df['prev_hit'] = ball_trajectory_df['hit'].shift(fill_value=0)
        #ball_trajectory_df['next_hit'] = ball_trajectory_df['hit'].shift(-1, fill_value=0)
        #ball_trajectory_df['hit'] = ball_trajectory_df[['hit', 'prev_hit', 'next_hit']].max(axis=1)

        drop_columns = ['Fast', 'Event', 'Z', 'Shot', 'player_X', 'player_Y', 'prev_hit', 'next_hit', 'Timestamp']
        
        ball_trajectory_df = ball_trajectory_df.drop(drop_columns, axis=1, errors='ignore')

        return ball_trajectory_df
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d = self.samples[idx]
        # Load images and convert them to tensors

        img = self.get_image_cache(d['cache_npy'])

        img = torch.from_numpy(img).float()
        target = torch.from_numpy(d['target'])

        img_files = [f"{self.root_dir}/../{im}" for im in d['img_files']]

        return {"img": img, "target": target, "img_files": img_files}

    def transform_coordinates(self, data, w, h, target_size=640):
        """
        Transform the X, Y coordinates in data based on image resizing and padding.
        
        Parameters:
        - data (torch.Tensor): A tensor of shape (N, 6) with columns (Frame, Visibility, X, Y, dx, dy).
        - w (int): Original width of the image.
        - h (int): Original height of the image.
        - target_size (int): The desired size for the longest side after resizing.
        
        Returns:
        - torch.Tensor: A transformed tensor of shape (N, 6).
        """
        
        # Clone the data to ensure we don't modify the original tensor in-place
        data_transformed = data
        
        # Determine padding
        max_dim = max(w, h)
        pad_diff = max_dim - min(w, h)
        pad1 = pad_diff // 2
        
        # Indices where x and y are not both 0
        indices_to_transform = (data[:, 2] != 0) | (data[:, 3] != 0)
        
        # Adjust for padding
        if h < w:
            data_transformed[indices_to_transform, 3] += pad1
        else:
            data_transformed[indices_to_transform, 2] += pad1  # if height is greater, adjust X

        # Adjust for scaling
        scale_factor = target_size / max_dim
        data_transformed[:, 2] *= scale_factor  # scale X
        data_transformed[:, 3] *= scale_factor  # scale Y
        data_transformed[:, 4] *= scale_factor  # scale dx
        data_transformed[:, 5] *= scale_factor  # scale dy
        
        return data_transformed
    def display_image_with_coordinates(self, img_tensor, coordinates):
        """
        Display an image with annotated coordinates.

        Parameters:
        - img_tensor (torch.Tensor): The image tensor of shape (C, H, W) or (H, W, C).
        - coordinates (list of tuples): A list of (X, Y) coordinates to be annotated.
        """
        
        # Convert the image tensor to numpy array
        img_array = img_tensor.permute(1, 2, 0).cpu().numpy()

        # Create a figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(img_array, cmap='gray')

        # Plot each coordinate
        for (x, y) in coordinates:
            ax.scatter(x, y, s=50, c='red', marker='o')
            ## Optionally, you can also draw a small rectangle around each point
            #rect = patches.Rectangle((x-5, y-5), 10, 10, linewidth=1, edgecolor='red', facecolor='none')
            #ax.add_patch(rect)

        plt.show()

    @lru_cache(maxsize=10)
    def __preprocess_img(self, path, pad_value=0):
        img = self.open_image(path)
        img = self.pad_to_square(img, pad_value)
        img = cv2.resize(img, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
        img.resize((1, 640, 640))
        return img

    def open_image(self, path):
        """Open image file, convert to grayscale and resize to half size
        """

        # Open the image file
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        
        # Reduce the resolution to half
        h, w = img.shape

        return cv2.resize(img, dsize=(w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
        
    def pad_to_square(self, img, pad_value=0):
        """Adjust 2D tensor to square by padding {pad_value}
        """
        h, w = img.shape
        dim_diff = np.abs(h - w)
        # (upper / left) padding and (lower / right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = (0, 0, pad1, pad2) if h > w else (pad1, pad2, 0, 0)
        # Add padding
        img = cv2.copyMakeBorder(img, *pad, borderType=cv2.BORDER_CONSTANT, value=pad_value)

        return img