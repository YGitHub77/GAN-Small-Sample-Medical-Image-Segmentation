# unet/discriminator.py
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=2, n_filters=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1),
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1),
            nn.BatchNorm2d(n_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_filters * 4, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)  # 拼接 (B,2,H,W)
        return self.model(x)


