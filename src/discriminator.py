from torch import nn

from src.config import D_WORD, D_GF


def get_img_encoder():
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


class Discriminator64(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_encoder = get_img_encoder()

    def forward(self, x):
        return self.img_encoder(x)
