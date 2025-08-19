import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage import binary_closing, binary_dilation
from scipy.ndimage import binary_opening

def preprocess_csvV5(csv_path, fps, head_width_px=20.0, duration_s=1/3):
    df_filtered = preprocess_csv_per_frame_motion_filter_with_padding_v2(
        csv_path, 13, fps, head_width_px, duration_s
    )
    
    # ===== 靜止點邏輯改為 erosion-dilation =====
    if 'static_ball' in df_filtered.columns:
        # 儲存原始 static_ball 供畫圖比較
        df_filtered['raw_static'] = df_filtered['static_ball'].astype(bool)

        static_mask = df_filtered['static_ball'].astype(bool).values
        kernel_size = int(fps * 0.2)
        kernel_size = max(kernel_size, 3)

        eroded = binary_erosion(static_mask, structure=np.ones(kernel_size))
        opened = binary_dilation(eroded, structure=np.ones(int(kernel_size)))

        df_filtered['static_ball'] = opened.astype(bool)

    # ===== 靜止點可視化 =====
    plot_static_comparison(
        df_filtered,
        save_path=convert_to_static_removal_path(csv_path)
    )

    # ===== 執行靜止點過濾 =====
    df_filtered.loc[df_filtered['static_ball'], 'Visibility'] = 0
    df_filtered.loc[df_filtered['static_ball'], ['X', 'Y']] = 0

    # ===== shift 處理與 event 處理 =====
    df_filtered['nX'] = df_filtered['X'].shift(-1).fillna(df_filtered['X'])
    df_filtered['nY'] = df_filtered['Y'].shift(-1).fillna(df_filtered['Y'])

    if 'Event' in df_filtered.columns:
        df_filtered['hit'] = ((df_filtered['Event'] == 1) | (df_filtered['Event'] == 2)).astype(int)
    else:
        df_filtered['hit'] = 0
    df_filtered.loc[df_filtered['static_ball'], 'hit'] = 0

    # ===== 清理欄位並存檔 =====
    df_filtered.to_csv(convert_to_static_removal_csv_path(csv_path, 'static_removal_before_csv'), index=False)

    df_filtered = df_filtered.drop(columns=[
        'static_ball', 'raw_static', 'motion_score', 'Fast', 'Event', 'Z', 'Shot',
        'player_X', 'player_Y', 'prev_hit', 'next_hit', 'Timestamp'
    ], errors='ignore')

    df_filtered.to_csv(convert_to_static_removal_csv_path(csv_path, 'static_removal_after_csv'), index=False)
    return df_filtered

def plot_static_comparison(df, save_path=None):
    has_ball = df['Visibility'] == 1
    raw_static = df.get('raw_static', pd.Series(False, index=df.index))
    filtered_static = df.get('static_ball', pd.Series(False, index=df.index))

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.invert_yaxis()

    # 有球軌跡
    plt.plot(df.loc[has_ball, 'X'], df.loc[has_ball, 'Y'], 
             label='Visible Trajectory', color='blue', alpha=0.6)

    # 原始 static（紅叉）
    # plt.scatter(df.loc[raw_static, 'X'], df.loc[raw_static, 'Y'], 
    #             color='red', marker='x', label='Original Static (raw)', zorder=5)

    # 開運算後保留的 static（橘圈）
    plt.scatter(df.loc[filtered_static, 'X'], df.loc[filtered_static, 'Y'], 
                facecolors='none', edgecolors='orange', marker='o',
                label='Final Static (filtered)', zorder=4, linewidths=1.5)

    plt.title("Static Point Filtering Comparison")
    plt.xlabel("X Position (px)")
    plt.ylabel("Y Position (px)")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def preprocess_csv(csv_file):
    # Read the ball_trajectory csv file
    ball_trajectory_df = pd.read_csv(csv_file)
    ball_trajectory_df['nX'] = ball_trajectory_df['X'].shift(-1).fillna(ball_trajectory_df['X'])
    ball_trajectory_df['nY'] = ball_trajectory_df['Y'].shift(-1).fillna(ball_trajectory_df['Y'])

    if 'Event' in ball_trajectory_df.columns:
        ball_trajectory_df['hit'] = ((ball_trajectory_df['Event'] == 1) | (ball_trajectory_df['Event'] == 2)).astype(int)
    else:
        ball_trajectory_df['hit'] = 0

    drop_columns = ['Fast', 'Event', 'Z', 'Shot', 'player_X', 'player_Y', 'prev_hit', 'next_hit', 'Timestamp']
    
    ball_trajectory_df = ball_trajectory_df.drop(drop_columns, axis=1, errors='ignore')

    return ball_trajectory_df

def compute_speed_no_smooth(df):
    speeds = [0.0]
    for i in range(1, len(df)):
        dx = df.loc[i, 'X'] - df.loc[i - 1, 'X']
        dy = df.loc[i, 'Y'] - df.loc[i - 1, 'Y']
        spd = math.sqrt(dx * dx + dy * dy)
        speeds.append(spd)
    return speeds

def filter_static_mask(df, speed_threshold=5.0, min_static_frames=5, static_radius=5):
    visible_df = df[df['Visibility'] > 0].copy()
    visible_df.reset_index(inplace=True)  # 保留原始 index
    if len(visible_df) == 0:
        return pd.Series([False] * len(df))  # 全部保留

    visible_df['speed'] = compute_speed_no_smooth(visible_df)
    visible_df['is_static'] = visible_df['speed'] < speed_threshold

    drop_indices = set()

    # 前端靜止段
    front_static_count = 0
    for is_static in visible_df['is_static']:
        if is_static:
            front_static_count += 1
        else:
            break

    if front_static_count >= min_static_frames:
        pivot = visible_df.iloc[0]
        for i in range(front_static_count):
            dist = math.hypot(visible_df.loc[i, 'X'] - pivot['X'], visible_df.loc[i, 'Y'] - pivot['Y'])
            if dist <= static_radius:
                drop_indices.add(visible_df.loc[i, 'index'])

    # 後端靜止段
    back_static_count = 0
    for is_static in reversed(visible_df['is_static'].tolist()):
        if is_static:
            back_static_count += 1
        else:
            break

    if back_static_count >= min_static_frames:
        pivot = visible_df.iloc[-1]
        for i in reversed(range(len(visible_df) - back_static_count, len(visible_df))):
            dist = math.hypot(visible_df.loc[i, 'X'] - pivot['X'], visible_df.loc[i, 'Y'] - pivot['Y'])
            if dist <= static_radius:
                drop_indices.add(visible_df.loc[i, 'index'])

    return df.index.isin(drop_indices)

def split_into_segments(df, max_missing_frames=30):
    segments = []
    start_idx = 0
    consecutive_missing = 0
    missing_run_start = -1

    for i in range(len(df)):
        vis = df.loc[i, 'Visibility']
        if vis == 0:
            if consecutive_missing == 0:
                missing_run_start = i
            consecutive_missing += 1
        else:
            if consecutive_missing >= max_missing_frames:
                segments.append(df.iloc[start_idx:missing_run_start].copy())
                start_idx = i
            consecutive_missing = 0

    if start_idx < len(df):
        segments.append(df.iloc[start_idx:].copy())

    return segments

def compute_motion_score(xy_seq, fps, head_width_px=20.0):
    xy = np.array(xy_seq)
    if np.any(np.isnan(xy)):
        return 0.0
    max_disp = np.max(np.linalg.norm(xy - xy[0], axis=1))
    
    # speeds = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    # mean_speed = np.mean(speeds)
    # coord_variance = np.var(xy[:, 0]) + np.var(xy[:, 1])

    # displacements = np.linalg.norm(xy[1:] - xy[:-1], axis=1)  # shape: (N-1,)
    # path_length = np.sum(displacements)
    # total_time = (len(xy) - 1) / fps
    # avg_speed = path_length / total_time
    # real_avg_seed_cm = avg_speed * 20 / head_width_px
    max_disp_cm = max_disp * 20 / head_width_px
    # motion_score = (
    #     0.8 * max_disp +
    #     1.0 * mean_speed +
    #     0.01 * coord_variance +    # 強縮放，避免爆炸
    #     0.8 * path_length
    # )
    return max_disp_cm

def preprocess_csv_per_frame_motion_filter_with_padding(
    csv_path,
    motion_score_threshold,
    fps,
    head_width_px,
    duration_s = 1/3,
):
    """
    幀級靜止球過濾版本（最終版）：對每一幀根據其周圍 window_size 幀計算 motion score，
    若低於 threshold 則將該幀 Visibility 改為 0。
    使用 padding 補齊邊界不足的幀。
    """
    df = pd.read_csv(csv_path)

    if 'Visibility' not in df.columns or 'X' not in df.columns or 'Y' not in df.columns:
        raise ValueError("CSV 欄位缺少必要資訊")

    window_size = int(duration_s*fps)

    half_w = window_size // 2
    min_visible_in_window = half_w
    motion_scores = np.zeros(len(df))
    removed_mask = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        start = max(i - half_w, 0)
        end = min(i + half_w, len(df))
        segment = df.iloc[start:end].copy()

        # 邊界補值（使用邊界幀複製填滿）
        if len(segment) < window_size:
            pad_len = window_size - len(segment)
            if i < half_w:
                pad_rows = pd.concat([segment.iloc[[0]]] * pad_len, ignore_index=True)
                segment = pd.concat([pad_rows, segment], ignore_index=True)
            else:
                pad_rows = pd.concat([segment.iloc[[-1]]] * pad_len, ignore_index=True)
                segment = pd.concat([segment, pad_rows], ignore_index=True)

        visible_segment = segment[segment['Visibility'] == 1]
        if len(visible_segment) < min_visible_in_window:
            continue
        xy_seq = list(zip(visible_segment['X'], visible_segment['Y']))
        motion_scores[i] = compute_motion_score(xy_seq, fps=fps, head_width_px=head_width_px)
        if motion_scores[i] < motion_score_threshold and df.loc[i, 'Visibility'] == 1:
            removed_mask[i] = True

    df['motion_score'] = motion_scores
    df['static_ball'] = removed_mask

    df = apply_segment_seeded_consistency(
        df,
        fps=fps,
        duration_s=duration_s,
        static_ratio_threshold=0.6,
        dynamic_ratio_threshold=0.4
    )

    return df

def preprocess_csv_per_frame_motion_filter_with_padding_v2(
    csv_path,
    motion_score_threshold,
    fps,
    head_width_px,
    duration_s = 1/3,
):
    """
    幀級靜止球過濾版本（最終版）：對每一幀根據其周圍 window_size 幀計算 motion score，
    若低於 threshold 則將該幀 Visibility 改為 0。
    使用 padding 補齊邊界不足的幀。
    """
    df = pd.read_csv(csv_path)

    if 'Visibility' not in df.columns or 'X' not in df.columns or 'Y' not in df.columns:
        raise ValueError("CSV 欄位缺少必要資訊")

    window_size = int(duration_s*fps)

    half_w = window_size // 2
    min_visible_in_window = half_w
    motion_scores = np.zeros(len(df))
    removed_mask = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        start = max(i - half_w, 0)
        end = min(i + half_w, len(df))
        segment = df.iloc[start:end].copy()

        # 邊界補值（使用邊界幀複製填滿）
        if len(segment) < window_size:
            pad_len = window_size - len(segment)
            if i < half_w:
                pad_rows = pd.concat([segment.iloc[[0]]] * pad_len, ignore_index=True)
                segment = pd.concat([pad_rows, segment], ignore_index=True)
            else:
                pad_rows = pd.concat([segment.iloc[[-1]]] * pad_len, ignore_index=True)
                segment = pd.concat([segment, pad_rows], ignore_index=True)

        visible_segment = segment[segment['Visibility'] == 1]
        if len(visible_segment) < min_visible_in_window:
            continue
        xy_seq = list(zip(visible_segment['X'], visible_segment['Y']))
        motion_scores[i] = compute_motion_score(xy_seq, fps=fps, head_width_px=head_width_px)
        if motion_scores[i] < motion_score_threshold and df.loc[i, 'Visibility'] == 1:
            removed_mask[i] = True

    df['motion_score'] = motion_scores
    df['static_ball'] = removed_mask

    return df

def apply_segment_seeded_consistency(df,
                                     fps,
                                     duration_s,
                                     static_ratio_threshold=0.6,
                                     dynamic_ratio_threshold=0.4):
    """
    雙向區段一致性處理：
    - 靜止點 seed：若 static_ratio >= 靜止閾值，整段設為靜止
    - 動態點 seed：若 static_ratio <= 動態閾值，整段取消靜止
    """
    window_size = int(duration_s * fps)
    is_static = df['static_ball'].values
    final_mask = np.array(is_static)
    N = len(df)

    i = 0
    while i < N:
        curr = is_static[i]

        if curr:
            start = i
            end = min(i + window_size, N)
            while end < N and is_static[end]:
                end += 1
            segment = is_static[start:end]
            static_ratio = np.mean(segment)

            if static_ratio >= static_ratio_threshold:
                final_mask[start:end] = True
                i = end
                continue
            else:
                final_mask[start:end] = False
                i = end
                continue
            # elif static_ratio <= dynamic_ratio_threshold:
            #     final_mask[start:end] = False
            #     i = end
            #     continue

        i += 1

    df['static_ball'] = final_mask
    return df

def preprocess_csvV4(csv_path, fps, head_width_px=20.0, duration_s=1/3):
    df_filtered = preprocess_csv_per_frame_motion_filter_with_padding(csv_path, 13, fps, head_width_px, duration_s)
    plot_visibility_removed_points_2d(df_filtered, save_path=convert_to_static_removal_path(csv_path))
    df_filtered.to_csv(convert_to_static_removal_csv_path(csv_path, 'static_removal_before_csv'), index=False)
    
    df_filtered.loc[df_filtered['static_ball'], 'Visibility'] = 0  # 執行靜止點過濾
    df_filtered.loc[df_filtered['static_ball'], 'X'] = 0  # 執行靜止點過濾
    df_filtered.loc[df_filtered['static_ball'], 'Y'] = 0  # 執行靜止點過濾

    df_filtered['nX'] = df_filtered['X'].shift(-1).fillna(df_filtered['X'])
    df_filtered['nY'] = df_filtered['Y'].shift(-1).fillna(df_filtered['Y'])

    if 'Event' in df_filtered.columns:
        df_filtered['hit'] = ((df_filtered['Event'] == 1) | (df_filtered['Event'] == 2)).astype(int)
    else:
        df_filtered['hit'] = 0
    df_filtered.loc[df_filtered['static_ball'], 'hit'] = 0  # 執行靜止點過濾

    df_filtered = df_filtered.drop(columns=['static_ball', 'motion_score',
                                            'Fast', 'Event', 'Z', 'Shot', 'player_X', 
                                            'player_Y', 'prev_hit', 'next_hit', 
                                            'Timestamp'], errors='ignore')
    df_filtered.to_csv(convert_to_static_removal_csv_path(csv_path, 'static_removal_after_csv'), index=False)
    
    return df_filtered

def is_static_shuttlecock(
    xy_seq,
    max_disp_threshold=25.0,
    mean_speed_threshold=3.0,
    variance_threshold=50.0,
    path_length_threshold=30.0
):
    """
    判斷一段 shuttlecock 是否為靜止狀態。

    Parameters:
    - xy_seq: List of (x, y), 長度固定為 10
    - *_threshold: 靜止判定門檻，單位為像素

    Returns:
    - True 表示判定為靜止球（可將 Visibility=1 改為 0）
    """
    xy = np.array(xy_seq)  # shape (10, 2)
    if np.any(np.isnan(xy)):
        return False

    max_disp = np.max(np.linalg.norm(xy - xy[0], axis=1))
    speeds = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    mean_speed = np.mean(speeds)
    coord_variance = np.var(xy[:, 0]) + np.var(xy[:, 1])
    path_length = np.sum(speeds)

    return (
        max_disp < max_disp_threshold and
        mean_speed < mean_speed_threshold and
        coord_variance < variance_threshold and
        path_length < path_length_threshold
    )

def preprocess_csv_static_filter_voting(
    csv_path,
    max_disp_threshold=25.0,
    mean_speed_threshold=3.0,
    variance_threshold=50.0,
    path_length_threshold=30.0,
    static_ratio_threshold=0.8,
    min_visible_frames=7,
    min_vote_support=3
):
    """
    處理整份 shuttlecock label CSV，根據移動特徵標記靜止段落為 static_ball=True。

    回傳處理後的 DataFrame（含 static_score、static_ball）。
    """
    df = pd.read_csv(csv_path)

    if 'Visibility' not in df.columns or 'X' not in df.columns or 'Y' not in df.columns:
        raise ValueError(f"{csv_path} 缺少必要欄位")
    df['nX'] = df['X'].shift(-1).fillna(df['X'])
    df['nY'] = df['Y'].shift(-1).fillna(df['Y'])

    if 'Event' in df.columns:
        df['hit'] = ((df['Event'] == 1) | (df['Event'] == 2)).astype(int)
    else:
        df['hit'] = 0

    vote_static = np.zeros(len(df), dtype=int)
    vote_total = np.zeros(len(df), dtype=int)

    for i in range(len(df) - 9):
        segment = df.iloc[i:i+10]
        if segment['Visibility'].sum() < min_visible_frames:
            continue

        xy_seq = list(zip(segment['X'], segment['Y']))
        if np.isnan(xy_seq).any():
            continue

        is_static = is_static_shuttlecock(
            xy_seq,
            max_disp_threshold=max_disp_threshold,
            mean_speed_threshold=mean_speed_threshold,
            variance_threshold=variance_threshold,
            path_length_threshold=path_length_threshold
        )

        for j in range(10):
            if i + j < len(df):
                vote_total[i + j] += 1
                if is_static:
                    vote_static[i + j] += 1

    vote_ratio = vote_static / (vote_total + 1e-5)
    static_flags = (vote_total >= min_vote_support) & (vote_ratio > static_ratio_threshold)

    df['static_score'] = vote_ratio
    df['static_ball'] = static_flags

    return df

def plot_visibility_removed_points_2d(df, save_path=None):
    original_visibility_mask = df['Visibility'] == 1
    removed_mask = original_visibility_mask & df['static_ball']

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.invert_yaxis()

    plt.plot(df.loc[original_visibility_mask, 'X'], 
             df.loc[original_visibility_mask, 'Y'], 
             label='Trajectory (Visibility=1)', color='blue', alpha=0.6)

    plt.scatter(df.loc[removed_mask, 'X'], 
                df.loc[removed_mask, 'Y'], 
                color='red', marker='x', label='Removed (V=1→0)', zorder=5)

    plt.title("Shuttlecock (X, Y) Trajectory - Static Points Removed")
    plt.xlabel("X Position (px)")
    plt.ylabel("Y Position (px)")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def preprocess_csvV3(csv_path):
    df_filtered = preprocess_csv_static_filter_voting(
        csv_path=csv_path)
    
    plot_visibility_removed_points_2d(df_filtered, save_path=convert_to_static_removal_path(csv_path))

    df_filtered.loc[df_filtered['static_ball'], 'hit'] = 0  # 執行靜止點過濾
    df_filtered.loc[df_filtered['static_ball'], 'Visibility'] = 0  # 執行靜止點過濾
    drop_columns = ['static_ball', 'static_score']
    df_filtered = df_filtered.drop(columns=drop_columns, errors='ignore')
    return df_filtered


def preprocess_csvV2(csv_path, speed_threshold=10.0, min_static_frames=5, max_missing_frames=20, static_radius=6.0):
    df_all = pd.read_csv(csv_path)

    if 'Visibility' not in df_all.columns:
        raise ValueError(f"CSV file 缺少 Visibility 欄位: {csv_path}")
    
    df_all['nX'] = df_all['X'].shift(-1).fillna(df_all['X'])
    df_all['nY'] = df_all['Y'].shift(-1).fillna(df_all['Y'])

    if 'Event' in df_all.columns:
        df_all['hit'] = ((df_all['Event'] == 1) | (df_all['Event'] == 2)).astype(int)
    else:
        df_all['hit'] = 0

    drop_columns = ['Fast', 'Event', 'Z', 'Shot', 'player_X', 'player_Y', 'prev_hit', 'next_hit', 'Timestamp']
    df_all = df_all.drop(drop_columns, axis=1, errors='ignore')

    df_all['is_removed'] = False
    df_all['segment_id'] = -1  # 預設 -1，未分段時

    segments = split_into_segments(df_all, max_missing_frames=max_missing_frames)

    for seg_id, seg in enumerate(segments):
        seg_idx = seg.index
        df_all.loc[seg_idx, 'segment_id'] = seg_id

        mask = filter_static_mask(seg, speed_threshold, min_static_frames, static_radius)
        df_all.loc[seg_idx[mask], 'is_removed'] = True

        df_all['Visibility_orig'] = df_all['Visibility']
        df_all.loc[seg_idx[mask], 'Visibility'] = 0

    plot_static_removal_comparison(df_all, convert_to_static_removal_path(csv_path))

    drop_columns = ['segment_id', 'is_removed', 'Visibility_orig']
    df_all = df_all.drop(drop_columns, axis=1, errors='ignore')

    return df_all

def convert_to_static_removal_path(csv_path):
    parent_dir = os.path.dirname(csv_path)                # /path/to/dir
    static_dir = os.path.join(os.path.dirname(parent_dir), 'static_removal')
    os.makedirs(static_dir, exist_ok=True)                # 自動建立目錄（若不存在）

    base_name = os.path.splitext(os.path.basename(csv_path))[0]  # filename
    return os.path.join(static_dir, f"{base_name}.png")

def convert_to_static_removal_csv_path(csv_path, filename):
    parent_dir = os.path.dirname(csv_path)                # /path/to/dir
    static_dir = os.path.join(os.path.dirname(parent_dir), filename)
    os.makedirs(static_dir, exist_ok=True)                # 自動建立目錄（若不存在）

    base_name = os.path.splitext(os.path.basename(csv_path))[0]  # filename
    return os.path.join(static_dir, f"{base_name}.csv")

def plot_static_removal_comparison(df, save_path=None):
    """
    視覺化每個 segment 的移動軌跡與被移除的靜止點：
    - 保留段連線 (線條 + 淡色點)
    - 靜止段標紅色叉叉
    """
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.invert_yaxis()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Static Removal Visualization")

    segment_ids = df['segment_id'].unique()
    cmap = plt.cm.get_cmap('tab10', len(segment_ids))

    for idx, seg_id in enumerate(segment_ids):
        seg = df[df['segment_id'] == seg_id]
        seg = seg[~((seg['X'] == 0) & (seg['Y'] == 0))]
        seg_kept = seg[~seg['is_removed']]
        seg_removed = seg[seg['is_removed']]

        color = cmap(idx)

        # 1. 畫折線（軌跡）
        plt.plot(seg_kept['X'], seg_kept['Y'],
                 color=color, alpha=0.8, linewidth=1.5,
                 label=f'Segment {seg_id} Trajectory')

        # 2. 保留點（淡色圓點）
        plt.scatter(seg_kept['X'], seg_kept['Y'],
                    color=color, alpha=0.4)

        # 3. 被移除點（紅色叉叉）
        plt.scatter(seg_removed['X'], seg_removed['Y'],
                    color='red', marker='x', label=f'Segment {seg_id} Removed')

    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"圖已儲存至: {save_path}")
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    # Example usage
    csv_file = '/Users/bartek/git/BartekTao/datasets/blion_tracknet_partial/csv/'
    # csv_file = '/Users/bartek/git/BartekTao/datasets/sportxai_2025/csv/'
    # csv_file = '/Users/bartek/git/BartekTao/datasets/sportxai_rally/csv/'

    # foreach read all csv files in the directory
    csv_files = [os.path.join(csv_file, f) for f in os.listdir(csv_file) if f.endswith('.csv')]
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        df = preprocess_csvV5(csv_file, fps=30, head_width_px=20.0, duration_s=1/3)
        # df = preprocess_csvV5(csv_file, fps=120, head_width_px=36.0, duration_s=1/2)
        # df = preprocess_csvV5(csv_file, fps=120, head_width_px=35.0, duration_s=1/3)