# coding:utf-8

import torch
from torch import nn


@torch.no_grad()
class AutoEncoder_31(nn.Module):
    def __init__(self, dim_mask=128, mask_size=96, temperature=1.):
        super().__init__()
        self.dim_mask = dim_mask
        self.mask_size = mask_size
        self.hidden_size = int(self.mask_size / 4)
        hidden_dim = int(4 * (self.mask_size / 4)**2)
        self.temperature = temperature

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=4, padding=1, bias=False),
            # nn.BatchNorm2d(16),
            nn.ReLU(True),  # mask_size / 4
            nn.Conv2d(16, 4, 3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(4),
            nn.ReLU(True),  # mask_size / 4
            nn.Flatten(),
            nn.Linear(hidden_dim, 256),
            # nn.LayerNorm(256),
            nn.ReLU(True),
            nn.Linear(256, self.dim_mask),
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(self.dim_mask, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(True),
            nn.Linear(1024, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(True),
        )

        self.decoder_conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),  # mask_size / 4
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),  # mask_size / 2
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),  # mask_size
            nn.Conv2d(8, 1, 3, stride=1, padding=1, bias=True)  # mask_size
        )

    def freeze_params(self):
        for m in self.modules():
            if not isinstance(m, nn.ReLU):
                m.eval()
        for p in self.parameters():
            p.requires_grad = False

    def encoding(self, x):

        assert x.shape[-1] == x.shape[-2] == self.mask_size, \
            print("The original mask_size of input should be equal to the supposed size.")

        x = self.encoder(x.unsqueeze(1))
        x /= self.temperature
        return x

    def decoding(self, x):

        assert x.shape[-1] == self.dim_mask
        x *= self.temperature
        x = self.decoder_fc(x)
        x = x.view(-1, 4, self.hidden_size, self.hidden_size)
        x = self.decoder_conv(x)

        return x


if __name__ == "__main__":
    x = torch.randn((2, 96, 96))
    net = AutoEncoder_31()
    net.freeze_params()
    x_embed = net.encoding(x)
    x_ = net.decoding(x_embed)
