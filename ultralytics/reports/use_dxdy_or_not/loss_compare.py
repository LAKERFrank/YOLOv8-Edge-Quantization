import os
import pandas as pd
import matplotlib.pyplot as plt

with_dxdy_file_path = './reports/use_dxdy_or_not/0908_200.csv'
without_dxdy_file_path = './reports/use_dxdy_or_not/0908_200_without_dxdy.csv'
result_folder = 'reports/use_dxdy_or_not/compare'
skiprows = 10
with_dxdy_df = pd.read_csv(with_dxdy_file_path, delimiter=',', skipinitialspace=True, on_bad_lines='skip', skiprows=range(1, skiprows+1))
without_dxdy_df = pd.read_csv(without_dxdy_file_path, delimiter=',', skipinitialspace=True, on_bad_lines='skip', skiprows=range(1, skiprows+1))

def plot_single_loss(df1, df2, loss_column, title, save_folder=None):
    plt.figure(figsize=(10, 6))
    plt.plot(df1['epoch'], df1[loss_column], label=f'with dxdy {loss_column}')
    plt.plot(df2['epoch'], df2[loss_column], label=f'without dxdy {loss_column}')
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

plot_single_loss(with_dxdy_df, without_dxdy_df, 'train/pos_loss', 'Pos Loss', result_folder)
plot_single_loss(with_dxdy_df, without_dxdy_df, 'train/conf_loss', 'Conf Loss', result_folder)
plot_single_loss(with_dxdy_df, without_dxdy_df, 'train/hit_loss', 'Hit Loss', result_folder)