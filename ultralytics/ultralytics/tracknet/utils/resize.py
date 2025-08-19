import cv2
import numpy as np

def resize_and_pad(path, target_size=640):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape[:2]
    
    # 計算縮放比例，保持長寬比
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 縮放圖片
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 創建一個 640x640 的黑色背景
    padded_image = np.zeros((target_size, target_size), dtype=np.uint8)
    
    # 計算置中位置
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    
    # 將縮放後的圖片放入黑色背景中
    padded_image[top:top+new_h, left:left+new_w] = resized
    
    return padded_image

# # 讀取圖片
# path = '/Users/bartek/git/BartekTao/datasets/tracknet/train_data/match_1/frame/1_00_01/0.png'
# im0 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# # 縮放並補滿
# output_image = resize_and_pad(path)

# # 顯示結果
# cv2.imshow('Padded Image', output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
