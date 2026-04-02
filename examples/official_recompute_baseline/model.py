import torch
import torch.nn as nn


device = "cuda" if torch.cuda.is_available() else "cpu"


class OfficialRecomputeMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=4096, depth=8, output_dim=1024):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        self.blocks = nn.ModuleList(layers)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)
