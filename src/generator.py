import torch
from torch import nn

from src.Attention import Attention
from src.util import upsample, residual_block

from src.config import D_Z, D_COND, D_GF, RESIDUALS, D_HIDDEN


class Generator0(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(D_Z + D_COND, D_GF * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(D_GF * 4 * 4 * 2),
            nn.modules.activation.GLU()
        )

        self.upsample_steps = nn.Sequential(*[upsample(D_GF // (2 ** i), D_GF // (2 ** (i + 1))) for i in range(4)])

    def forward(self, z_code, c_code):
        x = torch.cat((c_code, z_code), 1)
        x = self.fc(x)
        x = x.view(-1, D_GF, 4, 4)  # -> D_GF x 4 x 4
        x = self.upsample_steps(x)

        return x  # D_GF/16 x 64 x 64


class GeneratorN(nn.Module):
    def __init__(self):
        super().__init__()
        self.residuals = nn.Sequential(*[residual_block(D_GF * 2) for _ in range(RESIDUALS)])
        self.attn = Attention(D_GF, D_HIDDEN)
        self.upsample = upsample(D_GF * 2, D_GF)

    def forward(self, h_code, c_code, word_embs, mask):
        """
            h_code1(query):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(context): batch x cdf x sourceL (sourceL=seq_len)
            c_code1: batch x idf x queryL
            att1: batch x sourceL x queryL
        """
        self.attn.applyMask(mask)
        c_code, att = self.attn(h_code, word_embs)
        h_c_code = torch.cat((h_code, c_code), 1)
        out_code = self.residuals(h_c_code)
        out_code = self.upsample(out_code)  # D_GF/2 x 2in_size x 2in_size
        return out_code, att
