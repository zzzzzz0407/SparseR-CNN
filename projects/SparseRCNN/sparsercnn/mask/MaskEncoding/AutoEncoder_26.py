# coding:utf-8

import torch
from torch import nn
import torch.nn.functional as F


class Swish(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


@torch.no_grad()
class AutoEncoder_26(nn.Module):
    def __init__(self, dim_mask=128, mask_size=96, temperature=0.05):
        super().__init__()
        self.dim_mask = dim_mask
        self.mask_size = mask_size
        self.hidden_size = int(self.mask_size / 4)
        hidden_dim = int(1 * (self.mask_size / 4)**2)
        self.temperature = temperature

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            Swish(True),  # mask_size
            nn.Conv2d(4, 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            Swish(True),  # mask_size / 2
            nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            Swish(True),  # mask_size / 4
            nn.Conv2d(16, 1, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
            Swish(True),  # mask_size / 4
            nn.Flatten(),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            Swish(True),
            nn.Linear(256, self.dim_mask),
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(self.dim_mask, 256),
            nn.LayerNorm(256),
            Swish(True),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            Swish(True),
        )

        self.decoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            Swish(True),  # mask_size / 4
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            Swish(True),  # mask_size / 2
            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4),
            Swish(True),  # mask_size
            nn.Conv2d(4, 1, 3, stride=1, padding=1, bias=True)  # mask_size
        )

    def freeze_params(self):
        for m in self.modules():
            if not isinstance(m, Swish):
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
        x = x.view(-1, 1, self.hidden_size, self.hidden_size)
        x = self.decoder_conv(x)

        return x
