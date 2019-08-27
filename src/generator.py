import torch
from torch import nn
from torch.nn import functional as F

from src.attention import Attention
from src.config import D_GF, D_Z, D_COND, D_HIDDEN, RESIDUALS, DEVICE
from src.util import upsample_block, residual_block, conv3x3, count_params


class Generator0(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_gf = D_GF * 16
        self.fc = nn.Sequential(
            nn.Linear(D_Z + D_COND, self.d_gf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(self.d_gf * 4 * 4 * 2),
            nn.modules.activation.GLU(dim=1)
        )

        self.upsample_steps = nn.Sequential(
            *[upsample_block(self.d_gf // (2 ** i), self.d_gf // (2 ** (i + 1))) for i in range(4)]
        )

        p_trainable, p_non_trainable = count_params(self)
        print(f'Generator0 params: trainable {p_trainable} - non_trainable {p_non_trainable}')

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
        self.attn = Attention(D_GF, D_HIDDEN)
        self.upsample = upsample_block(D_GF * 2, D_GF)

        p_trainable, p_non_trainable = count_params(self)
        print(f'GeneratorN params: trainable {p_trainable} - non_trainable {p_non_trainable}')

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

        p_trainable, p_non_trainable = count_params(self)
        print(f'Image output params: trainable {p_trainable} - non_trainable {p_non_trainable}')

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

        return generated, attention, mu, logvar


class CondAug(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        self.fc = nn.Linear(D_HIDDEN, D_COND * 4, bias=True).to(device)

    def encode(self, text_emb):
        x = F.glu(self.fc(text_emb))
        mu = x[:, :D_COND]
        logvar = x[:, D_COND:]
        return mu, logvar

    def reparam(self, mu, logvar):
        std = logvar.mul(0.5).exp_().to(self.device)
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = nn.Parameter(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_emb):
        mu, logvar = self.encode(text_emb)
        c_code = self.reparam(mu, logvar)
        return c_code, mu, logvar
