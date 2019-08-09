import torch
from torch import nn
import torch.nn.functional as F


def roll_tensor(t, n, dim=0):
    return torch.cat((t[-n:], t[:-n]), dim=dim)


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.factor, mode=self.mode)


def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)


def upsample_block(in_planes, out_planes):
    return nn.Sequential(
        Interpolate(scale_factor=2, mode='nearest'),
        # nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        nn.modules.activation.GLU(dim=1)
    )


def residual_block(channels):
    return nn.Sequential(
        conv3x3(channels, channels * 2),
        nn.BatchNorm2d(channels * 2),
        nn.modules.activation.GLU(dim=1),
        conv3x3(channels, channels),
        nn.BatchNorm2d(channels)
    )
