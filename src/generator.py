import torch
from torch import nn
from src.util import upsample

from src.config import D_Z, D_COND, D_GF


class Generator0(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(D_Z + D_COND, D_GF * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(D_GF * 4 * 4 * 2),
            nn.modules.activation.GLU()
        )

        self.upsample_steps = [upsample(D_GF // (2 ** i), D_GF // (2 ** i)) for i in range(4)]

    def forward(self, z_code, c_code):
        x = torch.cat((c_code, z_code), 1)
        x = self.fc(x)
        x = x.view(-1, D_GF, 4, 4)  # -> D_GF x 4 x 4
        for u in self.upsample_steps:
            x = u(x)

        return x  # D_GF/16 x 64 x 64
