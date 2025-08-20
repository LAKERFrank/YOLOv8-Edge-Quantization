from torch import nn
import torch
import torch.nn.functional as F


class PreprocessLayer(nn.Module):
    def __init__(self, in_channels=30, num_frames=10):
        super().__init__()
        self.num_frames = num_frames

        # 灰階轉換：固定權重，不訓練
        self.to_grayscale = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.to_grayscale.weight.data = torch.tensor([[[[0.2989]], [[0.5870]], [[0.1140]]]])
        self.to_grayscale.weight.requires_grad = False

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.num_frames * 3

        # (B, 30, H, W) -> (B*10, 3, H, W)
        x = x.view(B * self.num_frames, 3, H, W)

        # 灰階轉換
        gray = self.to_grayscale(x)  # (B*10, 1, H, W)

        # reshape 回原始維度
        gray = gray.view(B, self.num_frames, H, W)  # (B, 10, H, W)

        # 背景
        bg = gray.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # 去背 + clip
        fg = F.relu(gray - bg)  # (B, 10, H, W)

        fg = fg / 255.0
        
        return fg

