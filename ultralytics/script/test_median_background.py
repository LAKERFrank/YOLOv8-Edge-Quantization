import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取 10 張圖片

file_paths = [
    "/Users/bartek/git/BartekTao/datasets/tracknet/train_data/match_2/frame/1_00_01/0.png", 
    "/Users/bartek/git/BartekTao/datasets/tracknet/train_data/match_2/frame/1_00_01/1.png",
    "/Users/bartek/git/BartekTao/datasets/tracknet/train_data/match_2/frame/1_00_01/2.png",
    "/Users/bartek/git/BartekTao/datasets/tracknet/train_data/match_2/frame/1_00_01/3.png",
    "/Users/bartek/git/BartekTao/datasets/tracknet/train_data/match_2/frame/1_00_01/4.png",
    "/Users/bartek/git/BartekTao/datasets/tracknet/train_data/match_2/frame/1_00_01/5.png",
    "/Users/bartek/git/BartekTao/datasets/tracknet/train_data/match_2/frame/1_00_01/6.png",
    "/Users/bartek/git/BartekTao/datasets/tracknet/train_data/match_2/frame/1_00_01/7.png",
    "/Users/bartek/git/BartekTao/datasets/tracknet/train_data/match_2/frame/1_00_01/8.png",
    "/Users/bartek/git/BartekTao/datasets/tracknet/train_data/match_2/frame/1_00_01/9.png",
    ]

frames = [cv2.imread(fp, cv2.IMREAD_GRAYSCALE) for fp in file_paths]


# 計算中位數影像
median_frame = np.median(frames, axis=0)

# 影像去除中位數
processed_frames = frames - median_frame

# 顯示部分原始影像、處理後影像和中位數影像
fig, axes = plt.subplots(3, 10, figsize=(15, 9))

for i in range(10):
    # 原始影像
    axes[0, i].imshow(frames[i], cmap='gray')
    axes[0, i].set_title(f'Original Frame {i+1}')
    axes[0, i].axis('off')

    # 去除中位數後的影像
    axes[1, i].imshow(processed_frames[i], cmap='gray')
    axes[1, i].set_title(f'Processed Frame {i+1}')
    axes[1, i].axis('off')

# 顯示計算出的中位數影像
axes[2, 2].imshow(median_frame, cmap='gray')
axes[2, 2].set_title('Median Frame')
axes[2, 2].axis('off')

plt.tight_layout()
plt.show()

for i, processed_frame in enumerate(processed_frames):
    output_path = f"/Users/bartek/git/BartekTao/datasets/test_background/processed_2_{i+1}.png"
    cv2.imwrite(output_path, processed_frame)
