from torch import nn

from src.config import D_WORD, D_GF
from src.util import conv3x3


def downscale16_encoder_block():
    return nn.Sequential(
        # in: BATCH x 3 x ih x iw
        # -> BATCH x D_GF x ih/2 x iw/2
        nn.Conv2d(in_channels=3, out_channels=D_GF, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # -> BATCH x D_GF*2 x ih/4 x iw/4
        nn.Conv2d(in_channels=D_GF, out_channels=D_GF * 2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(D_GF * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # -> BATCH x D_GF*4 x ih/8 x iw/8
        nn.Conv2d(in_channels=D_GF * 2, out_channels=D_GF * 4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(D_GF * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # -> BATCH x D_GF*8 x ih/16 x iw/16
        nn.Conv2d(in_channels=D_GF * 4, out_channels=D_GF * 8, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(D_GF * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )


def downscale2_encoder_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )


def conv3x3_LReLU(in_channels, out_channels):
    return nn.Sequential(
        conv3x3(in_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )


class Discriminator64(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = downscale16_encoder_block()

    def forward(self, x):
        return self.encoder(x)


class Discriminator128(nn.Module):
    def __init__(self):
        super().__init__()
        self.downscale_encoder_16 = downscale16_encoder_block()
        self.downscale_encoder_32 = downscale2_encoder_block(D_GF * 8, D_GF * 16)
        self.encoder32 = conv3x3_LReLU(D_GF * 16, D_GF * 8)

    def forward(self, x):
        x = self.downscale_encoder_16(x)  # -> BATCH x D_GF*8 x 8 x 8
        x = self.downscale_encoder_32(x)  # -> BATCH x D_GF*16 x 4 x 4
        x = self.encoder32(x)  # -> BATCH x D_GF*8 x 4 x 4
        return x


class Discriminator256(nn.Module):
    def __init__(self):
        super().__init__()
        self.downscale_encoder_16 = downscale16_encoder_block()
        self.downscale_encoder_32 = downscale2_encoder_block(D_GF * 8, D_GF * 16)
        self.downscale_encoder_64 = downscale2_encoder_block(D_GF * 16, D_GF * 32)
        self.encoder64 = conv3x3_LReLU(D_GF * 32, D_GF * 16)
        self.encoder64_2 = conv3x3_LReLU(D_GF * 16, D_GF * 8)

    def forward(self, x):
        x = self.downscale_encoder_16(x)
        x = self.downscale_encoder_32(x)
        x = self.downscale_encoder_64(x)
        x = self.encoder64(x)
        x = self.encoder64_2(x)
        return x

