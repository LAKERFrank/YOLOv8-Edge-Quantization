import os
import glob
import math
import pandas as pd
import matplotlib.pyplot as plt

def split_into_segments(df, max_missing_frames=30):
    """
    將整個 DataFrame 依據 'Visibility' 連續為 0 的段落切割成多個 segments。
    當連續 >= max_missing_frames 幀為不可見時，視為一段的結束。
    回傳一個 list，每個元素為一個 segment 的 DataFrame。
    """
    segments = []
    start_idx = 0
    consecutive_missing = 0
    missing_run_start = -1

    for i in range(len(df)):
        vis = df.loc[i, 'Visibility']
        if vis == 0:
            # 累計連續不可見幀數
            if consecutive_missing == 0:
                missing_run_start = i
            consecutive_missing += 1
        else:
            # 一旦遇到可見，檢查之前是否累計超過門檻
            if consecutive_missing >= max_missing_frames:
                # 代表上一段結束於 (missing_run_start - 1)
                # 建立一個 segment
                segment = df.iloc[start_idx : missing_run_start].copy()
                segments.append(segment)
                # 新的 segment 從這個可見幀開始
                start_idx = i
            consecutive_missing = 0

    # 若最後還有剩餘 frames
    if start_idx < len(df):
        segment = df.iloc[start_idx:].copy()
        segments.append(segment)

    return segments

def smooth_positions(df, smoothing_window=5):
    
    # 進行移動平均平滑
    df['x_smooth'] = df['X'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
    df['y_smooth'] = df['Y'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
    
    return df


def compute_speed(df):
    """
    根據平滑後的 (x_smooth, y_smooth) 計算相鄰 frame 的速度
    """
    speeds = [0.0]
    for i in range(1, len(df)):
        dx = df.loc[i, 'x_smooth'] - df.loc[i-1, 'x_smooth']
        dy = df.loc[i, 'y_smooth'] - df.loc[i-1, 'y_smooth']
        spd = math.sqrt(dx*dx + dy*dy)
        speeds.append(spd)
    return speeds

def filter_static_segments(df, speed_threshold=5.0, min_static_frames=5, smoothing_window=5, static_radius=5):
    """
    1. 先只保留 Visibility > 0 的 frame
    2. 平滑處理與計算速度，標記速度小的 frame 為 is_static
    3. 依照原邏輯抓出候選靜止區段（連續 is_static）
       接著以候選區段的 pivot (前端取最後一筆、後端取第一筆) 為基準，
       檢查候選區段內連續 frame 與 pivot 的歐式距離是否均在 static_radius 內，
       若達到 min_static_frames 才視為真正靜止並移除該區段。
    回傳: (filtered_df, original_df)
    """
    # 先篩除 Visibility 為 0 的 frame
    df = df[df['Visibility'] > 0].copy()
    df.reset_index(drop=True, inplace=True)
    if len(df) == 0:
        return df, df

    # 平滑與計算速度
    df = smooth_positions(df, smoothing_window)
    df['speed'] = compute_speed(df)
    df['is_static'] = df['speed'] < speed_threshold

    # 前端候選區段
    front_static_count = 0
    for is_static in df['is_static']:
        if is_static:
            front_static_count += 1
        else:
            break

    new_front_count = 0
    if front_static_count > 0:
        # 選擇前端候選區段最後一筆作為 pivot
        pivot_front = df.iloc[0]
        once = True
        for i in range(front_static_count):
            dist = math.sqrt((df.loc[i, 'X'] - pivot_front['X'])**2 + (df.loc[i, 'Y'] - pivot_front['Y'])**2)
            if dist <= static_radius:
                new_front_count += 1
            else:
                if once and i < front_static_count/2:
                    pivot_front = df.iloc[i]
                    once = False
                else:
                    break

    # 後端候選區段
    back_static_count = 0
    for is_static in reversed(df['is_static'].tolist()):
        if is_static:
            back_static_count += 1
        else:
            break

    new_back_count = 0
    if back_static_count > 0:
        # 選擇後端候選區段第一筆作為 pivot
        pivot_back = df.iloc[len(df) - 1]
        once = True
        for i in reversed(range(len(df) - back_static_count, len(df))):
            dist = math.sqrt((df.loc[i, 'X'] - pivot_back['X'])**2 + (df.loc[i, 'Y'] - pivot_back['Y'])**2)
            if dist <= static_radius:
                new_back_count += 1
            else:
                if once and i > back_static_count/2:
                    pivot_back = df.iloc[i]
                    once = False
                else:
                    break

    # 判斷是否滿足 min_static_frames 的要求，否則不移除
    start_idx = new_front_count if front_static_count >= min_static_frames else 0
    end_idx = len(df) - new_back_count if back_static_count >= min_static_frames else len(df)

    filtered_df = df.iloc[start_idx:end_idx].copy()
    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df, df

def main():
    input_folder = r'/Users/bartek/git/BartekTao/datasets/test_static_ball'
    speed_threshold = 10.0
    min_static_frames = 5
    smoothing_window = 1
    max_missing_frames = 20
    static_radius = 6.0

    csv_files = glob.glob(os.path.join(input_folder, "*_ball.csv"))
    for csv_file in csv_files:
        df_all = pd.read_csv(csv_file)
        # 若無 Visibility 欄位可自行調整
        if 'Visibility' not in df_all.columns:
            print(f"檔案 {csv_file} 缺少 Visibility 欄位，跳過處理。")
            continue

        # Step 1: 將整場資料依據長時間不可見切割成多個 segments
        segments = split_into_segments(df_all, max_missing_frames=max_missing_frames)

        # Step 2: 對每個 segment 做靜止段去除
        filtered_segments = []
        original_segments = []

        for seg_id, seg_df in enumerate(segments):
            filtered_df, original_df = filter_static_segments(
                seg_df,
                speed_threshold=speed_threshold,
                min_static_frames=min_static_frames,
                smoothing_window=smoothing_window,
                static_radius=static_radius
            )
            # 在這裡也可以幫 filtered_df 加上 segment_id
            filtered_df['segment_id'] = seg_id
            original_df['segment_id'] = seg_id

            filtered_segments.append(filtered_df)
            original_segments.append(original_df)

        # 合併所有 segment
        final_filtered = pd.concat(filtered_segments, ignore_index=True)
        final_original = pd.concat(original_segments, ignore_index=True)

        # 輸出檔案
        base_name, ext = os.path.splitext(csv_file)
        output_csv = f"{base_name}_filtered{ext}"
        final_filtered.to_csv(output_csv, index=False)

        # 視覺化 (可視情況選擇分 segment 畫或全部一起畫)
        output_png = f"{base_name}_comparison.png"
        plot_segments(final_original, final_filtered, output_png)

        print(f"處理完畢: {csv_file}")
        print(f"  -> 分段數量: {len(segments)}")
        print(f"  -> 過濾後結果: {output_csv}")
        print(f"  -> 視覺化圖檔: {output_png}\n")

def plot_segments(original_df, filtered_df, save_path):
    """
    繪製多段資料在同一張圖上，並區分 segment_id
    原點置於左上角，即 y 軸反轉
    """

    plt.figure(figsize=(8,6))
    ax = plt.gca()
    plt.title(os.path.basename(save_path))
    plt.xlabel('x')
    plt.ylabel('y')
    
    # 反轉 y 軸，使原點在左上角
    ax.invert_yaxis()

    # 依 segment_id 分顏色
    segment_ids = original_df['segment_id'].unique()
    colors = plt.cm.get_cmap('tab10', len(segment_ids))

    for idx, seg_id in enumerate(segment_ids):
        # 取出該 segment 的原始
        seg_original = original_df[original_df['segment_id'] == seg_id]
        # 取出該 segment 的最終保留
        seg_filtered = filtered_df[filtered_df['segment_id'] == seg_id]

        # 畫原始(可見)座標
        plt.scatter(seg_original['X'], seg_original['Y'],
                    color=colors(idx), alpha=0.3,
                    label=f'Seg{seg_id} Original' if idx==0 else None)
        # 若有平滑欄位
        if 'x_smooth' in seg_original.columns and len(seg_original) > 1:
            plt.plot(seg_original['x_smooth'], seg_original['y_smooth'],
                     color=colors(idx), alpha=0.5,
                     label=f'Seg{seg_id} Smoothed' if idx==0 else None)
        # 畫過濾後
        plt.scatter(seg_filtered['X'], seg_filtered['Y'],
                    color=colors(idx), marker='x',
                    label=f'Seg{seg_id} Filtered' if idx==0 else None)

    plt.legend()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
