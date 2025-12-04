import torch
from torch import nn
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    """U-Net Generator with skip connections for M -> S mapping"""

    def __init__(self, segment_length=8192):
        super().__init__()
        self.segment_length = segment_length

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder with skip connections
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(512, 128, kernel_size=4, stride=2, padding=1),  # 256+256 from skip
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, kernel_size=4, stride=2, padding=1),  # 128+128 from skip
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=15, padding=7),  # 64+64 from skip
            nn.Tanh(),
        )

    def forward(self, m):
        x = m.unsqueeze(1)  # [B, 1, T]

        # Encoder
        e1 = self.enc1(x)    # [B, 64, T]
        e2 = self.enc2(e1)   # [B, 128, T/2]
        e3 = self.enc3(e2)   # [B, 256, T/4]
        e4 = self.enc4(e3)   # [B, 512, T/8]

        # Decoder with skip connections
        d4 = self.dec4(e4)   # [B, 256, T/4]
        d3 = self.dec3(torch.cat([d4, e3], dim=1))  # [B, 128, T/2]
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # [B, 64, T]
        out = self.dec1(torch.cat([d2, e1], dim=1)) # [B, 1, T]

        return out.squeeze(1)  # [B, T]


class Discriminator(nn.Module):
    """PatchGAN Discriminator with spectral normalization"""

    def __init__(self, segment_length=8192):
        super().__init__()
        self.segment_length = segment_length

        self.features = nn.Sequential(
            spectral_norm(nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(256, 512, kernel_size=15, stride=2, padding=7)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        reduced_length = max(1, segment_length // 16)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            spectral_norm(nn.Linear(512 * reduced_length, 1)),
        )

    def forward(self, s):
        x = s.unsqueeze(1)  # [B, 1, T]
        out = self.features(x)
        out = self.classifier(out)
        return out
