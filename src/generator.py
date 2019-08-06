import torch
from torch import nn

from src.attention import Attention
from src.encoder import CondAug
from src.util import upsample_block, residual_block, conv3x3

from src.config import D_Z, D_COND, D_GF, RESIDUALS, D_HIDDEN, D_WORD


class Generator0(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_gf = D_GF * 16
        self.fc = nn.Sequential(
            nn.Linear(D_Z + D_COND, self.d_gf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(self.d_gf * 4 * 4 * 2),
            nn.modules.activation.GLU()
        )

        self.upsample_steps = nn.Sequential(
            *[upsample_block(self.d_gf // (2 ** i), self.d_gf // (2 ** (i + 1))) for i in range(4)])

    def forward(self, z_code, c_code):
        x = torch.cat((c_code, z_code), 1)
        x = self.fc(x)
        x = x.view(-1, self.d_gf, 4, 4)  # -> D_GF x 4 x 4
        x = self.upsample_steps(x)

        return x  # D_GF/16 x 64 x 64


class GeneratorN(nn.Module):
    def __init__(self):
        super().__init__()
        self.residuals = nn.Sequential(*[residual_block(D_GF * 2) for _ in range(RESIDUALS)])
        self.attn = Attention(D_GF, D_WORD)
        self.upsample = upsample_block(D_GF * 2, D_GF)

    def forward(self, h_code, c_code, word_embs, mask):
        """
            h_code1(query), output of previous generator:  batch x D_GF x ih x iw (queryL=ihxiw)
            word_embs(context): batch x D_COND x seq_len
            c_code1: batch x D_GF x ih x iw
            att1: batch x sourceL x ih x iw
        """
        self.attn.applyMask(mask)
        c_code, att = self.attn(h_code, word_embs)
        h_c_code = torch.cat((h_code, c_code), 1)
        out_code = self.residuals(h_c_code)
        out_code = self.upsample(out_code)  # D_GF/2 x 2in_size x 2in_size
        return out_code, att


class ImageGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.img = nn.Sequential(
            conv3x3(D_GF, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        return self.img(h_code)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_aug = CondAug()
        self.f0 = Generator0()
        self.f1 = GeneratorN()
        self.f2 = GeneratorN()
        self.img0 = ImageGen()
        self.img1 = ImageGen()
        self.img2 = ImageGen()

    def forward(self, z_code, sent_emb, word_embs, mask):
        generated = []
        attention = []

        c_code, mu, logvar = self.cond_aug(sent_emb)

        h1 = self.f0(z_code, c_code)
        generated.append(self.img0(h1))

        h2, a1 = self.f1(h1, c_code, word_embs, mask)
        generated.append(self.img1(h2))
        attention.append(a1)

        h3, a2 = self.f2(h2, c_code, word_embs, mask)
        generated.append(self.img2(h3))
        attention.append(a2)

        return generated
