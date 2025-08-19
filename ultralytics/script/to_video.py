import cv2
import os

def images_to_video(image_folder, output_video, fps=30, scale=1.5):
    # 取得所有照片檔案並排序
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(os.path.splitext(x)[0]))

    # 讀取第一張照片取得其大小
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    
    # 計算放大後的大小
    height, width, _ = frame.shape
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 建立影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 編碼器
    video = cv2.VideoWriter(output_video, fourcc, fps, (new_width, new_height))

    # 將每張照片放大並寫入影片
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)

        # 放大圖片
        frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        video.write(frame_resized)  # 寫入放大後的影格

    # 釋放資源
    video.release()
    print(f"放大影片已儲存至 {output_video}")

# 使用範例
path = '/Users/bartek/git/BartekTao/tracknet_report/train348_val_img/pred_red'
images_to_video(path, f'{path}/output_video.mp4', fps=120, scale=1.5)
