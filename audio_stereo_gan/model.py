import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, input_dim=16384, output_dim=16384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.LeakyReLU(0.2),
            nn.Linear(4096, output_dim),
            nn.Tanh()
        )

    def forward(self, m):
        return self.net(m)


class Discriminator(nn.Module):
    def __init__(self, input_dim=16384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.LeakyReLU(0.2),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward(self, s):
        return self.net(s)
