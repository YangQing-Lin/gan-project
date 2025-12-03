import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, segment_length=8192):
        super().__init__()
        self.segment_length = segment_length

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 1, kernel_size=15, padding=7),
            nn.Tanh(),
        )

    def forward(self, m):
        x = m.unsqueeze(1)  # [B, 1, T]
        out = self.encoder(x)
        out = self.decoder(out)
        return out.squeeze(1)  # [B, T]


class Discriminator(nn.Module):
    def __init__(self, segment_length=8192):
        super().__init__()
        self.segment_length = segment_length

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        reduced_length = max(1, segment_length // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * reduced_length, 1),
            nn.Sigmoid()
        )

    def forward(self, s):
        x = s.unsqueeze(1)  # [B, 1, T]
        out = self.features(x)
        out = self.classifier(out)
        return out
