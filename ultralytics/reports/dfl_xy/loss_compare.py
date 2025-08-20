import os
import pandas as pd
import matplotlib.pyplot as plt

without_iou_file_path = './reports/dfl_xy/0920_200_dfl_only_xy.csv'
with_iou_file_path = './reports/dfl_xy/0920_200_dfl_dxdx_iou.csv'
result_folder = 'reports/dfl_xy/compare'
skiprows = 25
without_iou_df = pd.read_csv(without_iou_file_path, delimiter=',', skipinitialspace=True, on_bad_lines='skip', skiprows=range(1, skiprows+1))
with_iou_df = pd.read_csv(with_iou_file_path, delimiter=',', skipinitialspace=True, on_bad_lines='skip', skiprows=range(1, skiprows+1))

def plot_single_loss(df1, df2, loss_column, title, save_folder=None):
    plt.figure(figsize=(10, 6))
    plt.plot(df1['epoch'], df1[loss_column], label=f'without iou {loss_column}')
    plt.plot(df2['epoch'], df2[loss_column], label=f'with iou {loss_column}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title} vs Epoch')
    plt.legend()
    plt.grid(True)
    
    # Save the plot if save_folder is specified
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)  # Create folder if it doesn't exist
        file_path = os.path.join(save_folder, f'{title}_vs_Epoch.png')
        plt.savefig(file_path)
        print(f'Plot saved at: {file_path}')
    else:
        plt.show()

plot_single_loss(without_iou_df, with_iou_df, 'train/pos_loss', 'Pos Loss', result_folder)
plot_single_loss(without_iou_df, with_iou_df, 'train/conf_loss', 'Conf Loss', result_folder)