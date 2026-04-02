"""
极简 2 层 MLP，用于 IR 捕获演示。
去除 LayerNorm 和 Dropout，生成的 ATen IR 短到可以逐行阅读。
"""
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleMLP(nn.Module):
    """Linear → ReLU → Linear"""

    def __init__(self, input_dim=64, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
