import torch
from torch import nn


class QNetwork(nn.Module):
    """DQN convolutional encoder for the 64x64 RGB Crafter observation."""

    def __init__(self, actions: int = 17):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            width = self.encoder(torch.zeros(1, 3, 64, 64)).shape[-1]
        self.head = nn.Sequential(nn.Linear(width, 512), nn.ReLU(), nn.Linear(512, actions))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.permute(0, 3, 1, 2).float().div(255)
        return self.head(self.encoder(x))
