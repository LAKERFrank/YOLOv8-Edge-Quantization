import os
import pandas as pd
import matplotlib.pyplot as plt

file_path = './reports/dfl_xy/0920_200_dfl_dxdx_iou.csv'
result_folder = 'reports/dfl_xy/0920_200_dfl_dxdx_iou'
## df = pd.read_csv(file_path, delimiter=',', skipinitialspace=True, on_bad_lines='skip')
skiprows = 25
df = pd.read_csv(file_path, delimiter=',', skipinitialspace=True, on_bad_lines='skip', skiprows=range(1, skiprows+1))

def plot_single_loss(df, loss_column, title, save_folder=None):
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df[loss_column], label=loss_column)
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

# Function to plot all losses and save the image
def plot_all_losses(df, save_folder=None):
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train/pos_loss'], label='Pos Loss')
    plt.plot(df['epoch'], df['train/conf_loss'], label='Conf Loss')
    plt.plot(df['epoch'], df['train/iou_loss'], label='IOU Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('All Losses vs Epoch')
    plt.legend()
    plt.grid(True)
    
    # Save the plot if save_folder is specified
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)  # Create folder if it doesn't exist
        file_path = os.path.join(save_folder, 'All_Losses_vs_Epoch.png')
        plt.savefig(file_path)
        print(f'Plot saved at: {file_path}')
    else:
        plt.show()

plot_single_loss(df, 'train/pos_loss', 'Pos Loss', result_folder)
plot_single_loss(df, 'train/conf_loss', 'Conf Loss', result_folder)
plot_single_loss(df, 'train/iou_loss', 'IOU Loss', result_folder)
plot_all_losses(df, result_folder)