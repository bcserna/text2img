import torch
from torch import nn

from src.config import D_DF, D_HIDDEN, DEVICE
from src.util import conv3x3, count_params


def downscale16_encoder_block():
    return nn.Sequential(
        # in: BATCH x 3 x ih x iw
        # -> BATCH x D_DF x ih/2 x iw/2
        nn.Conv2d(in_channels=3, out_channels=D_DF, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # -> BATCH x D_DF*2 x ih/4 x iw/4
        nn.Conv2d(in_channels=D_DF, out_channels=D_DF * 2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(D_DF * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # -> BATCH x D_DF*4 x ih/8 x iw/8
        nn.Conv2d(in_channels=D_DF * 2, out_channels=D_DF * 4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(D_DF * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # -> BATCH x D_DF*8 x ih/16 x iw/16
        nn.Conv2d(in_channels=D_DF * 4, out_channels=D_DF * 8, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(D_DF * 8),
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


class DiscriminatorLogitBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_logit = CondLogitBlock()
        self.uncond_logit = LogitBlock()

    def forward(self, h, condition=None):
        if condition is not None:
            return self.cond_logit(h, condition)
        return self.uncond_logit(h)


class CondLogitBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.joint_conv = conv3x3_LReLU(D_DF * 8 + D_HIDDEN, D_DF * 8)
        self.logit = LogitBlock()

    def forward(self, h, cond):
        cond = cond.view(-1, D_HIDDEN, 1, 1)
        cond = cond.repeat(1, 1, h.size(-2), h.size(-1))
        cond_h = torch.cat((h, cond), 1)
        cond_h = self.joint_conv(cond_h)
        logits = self.logit(cond_h)
        return logits


class LogitBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Conv2d(D_DF * 8, 1, kernel_size=4, stride=4)

    def forward(self, h):
        return self.logits(h)


class Discriminator64(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = downscale16_encoder_block()
        self.logit = DiscriminatorLogitBlock()

        p_trainable, p_non_trainable = count_params(self)
        print(f'Discriminator64 params: trainable {p_trainable} - non_trainable {p_non_trainable}')

    def forward(self, x):
        return self.encoder(x)


class Discriminator128(nn.Module):
    def __init__(self):
        super().__init__()
        self.downscale_encoder_16 = downscale16_encoder_block()
        self.downscale_encoder_32 = downscale2_encoder_block(D_DF * 8, D_DF * 16)
        self.encoder32 = conv3x3_LReLU(D_DF * 16, D_DF * 8)
        self.logit = DiscriminatorLogitBlock()

        p_trainable, p_non_trainable = count_params(self)
        print(f'Discriminator128 params: trainable {p_trainable} - non_trainable {p_non_trainable}')

    def forward(self, x):
        x = self.downscale_encoder_16(x)  # -> BATCH x D_DF*8 x 8 x 8
        x = self.downscale_encoder_32(x)  # -> BATCH x D_DF*16 x 4 x 4
        x = self.encoder32(x)  # -> BATCH x D_DF*8 x 4 x 4
        return x


class Discriminator256(nn.Module):
    def __init__(self):
        super().__init__()
        self.downscale_encoder_16 = downscale16_encoder_block()
        self.downscale_encoder_32 = downscale2_encoder_block(D_DF * 8, D_DF * 16)
        self.downscale_encoder_64 = downscale2_encoder_block(D_DF * 16, D_DF * 32)
        self.encoder64 = conv3x3_LReLU(D_DF * 32, D_DF * 16)
        self.encoder64_2 = conv3x3_LReLU(D_DF * 16, D_DF * 8)
        self.logit = DiscriminatorLogitBlock()

        p_trainable, p_non_trainable = count_params(self)
        print(f'Discriminator256 params: trainable {p_trainable} - non_trainable {p_non_trainable}')

    def forward(self, x):
        x = self.downscale_encoder_16(x)  # -> BATCH x D_DF*8 x 16 x 16
        x = self.downscale_encoder_32(x)  # -> BATCH x D_DF*16 x 8 x 8
        x = self.downscale_encoder_64(x)  # -> BATCH x D_DF*32 x 4 x 4
        x = self.encoder64(x)  # -> BATCH x D_DF*16 x 4 x 4
        x = self.encoder64_2(x)  # -> BATCH x D_DF*8 x 4 x 4
        return x


class Discriminator(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        self.d64 = Discriminator64().to(self.device)
        self.d128 = Discriminator128().to(self.device)
        self.d256 = Discriminator256().to(self.device)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def forward(self, x):
        o64 = self.d64(x[0].to(self.device))
        o128 = self.d128(x[1].to(self.device))
        o256 = self.d256(x[2].to(self.device))
        return o64, o128, o256

    def get_logits(self, x, condition):
        l64 = self.d64.logit(x[0], condition)
        l128 = self.d128.logit(x[1], condition)
        l256 = self.d256.logit(x[2], condition)
        return l64, l128, l256
