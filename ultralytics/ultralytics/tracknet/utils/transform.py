
import math
import torch


def target_grid(target_x, target_y, stride, target_shape = 640):
    grid_x = int(target_x / stride)
    grid_y = int(target_y / stride)
    offset_x = (target_x % stride)
    offset_y = (target_y % stride)

    limit_grid_size = int(target_shape / stride)
    adjust_offset = int(stride - 1)
    if grid_x == limit_grid_size:
        grid_x = limit_grid_size - 1
        offset_x = adjust_offset
    if grid_y == limit_grid_size:
        grid_y = limit_grid_size - 1
        offset_y = adjust_offset
    assert grid_x < limit_grid_size and grid_y < limit_grid_size
    return grid_x, grid_y, offset_x, offset_y

def calculate_dist(p1, p2):
    # 計算兩點之間的距離
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def calculate_angle(p1, p2, p3):
    # 計算從p1到p2和從p2到p3之間的夾角
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    
    # 計算內積和模長
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    # 如果其中一個向量為零，則無法計算角度
    if mag_v1 == 0 or mag_v2 == 0:
        return None  # 或者返回特定值，根據需要設置
    
    # 限制範圍避免數值問題
    cosine_value = max(-1, min(1, dot_product / (mag_v1 * mag_v2)))

    # 計算角度（弧度），並轉換為角度
    angle_rad = math.acos(cosine_value)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg


# 一次10張照片, cell = 20*20
# output: [](ball_cell_xy, conf)
def decode_pred_conf(pred_confs, threshold=0.8):
    # 將輸入的預測值轉為機率
    pred_probs = torch.sigmoid(pred_confs)

    # 將 pred_probs 轉換為形狀 (10, 20, 20)
    pred_scores = pred_probs.view(10, 20, 20)

    # 保存結果
    results = []

    # 迭代每個 frame (10 張圖片)
    for frame_idx in range(10):
        # 獲取當前圖片的 conf
        p_conf = pred_scores[frame_idx]

        # 獲取大於 threshold 的位置及其值
        indices = torch.nonzero(p_conf > threshold, as_tuple=True)
        values = p_conf[indices]

        # 將 indices (y, x) 轉換為 (cell_y, cell_x)
        cells = list(zip(indices[0].tolist(), indices[1].tolist()))

        # 將 (cell, value) 封裝到結果中
        frame_results = [(cell, value.item()) for cell, value in zip(cells, values)]
        results.append(frame_results)

    return results

def inverse_transform(pred_x, pred_y, target_weight, target_hight):
    return


def revert_coordinates(data, w=1280, h=480, target_size=640):
    """
    Reverts coordinates from resized-padded image (e.g., 640x640)
    back to original image coordinates (e.g., 1280x480).
    
    Parameters:
    - data: (N, 6) tensor with (frame, visibility, x, y, dx, dy)
    - w, h: original image size
    - target_size: size after preprocessing
    
    Returns:
    - reverted (N, 6) tensor in original image coordinates
    """
    data_reverted = data.clone()

    max_dim = max(w, h)
    scale_factor = target_size / max_dim
    pad_diff = max_dim - min(w, h)
    pad1 = pad_diff // 2  # = (1280 - 480) / 2 = 400 → /2 = 200

    indices = (data[:, 2] != 0) | (data[:, 3] != 0)

    # Remove padding
    if h < w:
        data_reverted[indices, 1] -= pad1
    else:
        data_reverted[indices, 0] -= pad1

    # Revert scaling
    data_reverted[:, 0] /= scale_factor
    data_reverted[:, 1] /= scale_factor

    return data_reverted