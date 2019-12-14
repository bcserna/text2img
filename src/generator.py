import torch
from torch import nn
from torch.nn import functional as F

from src.attention import Attention
from src.config import D_GF, D_Z, D_COND, D_HIDDEN, RESIDUALS, DEVICE
from src.util import upsample_block, conv3x3, count_params, Residual


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


def self_attn_block():
    return nn.Sequential(
        nn.Conv2d(D_GF, D_GF, 4, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(D_GF, D_GF, 4, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        SelfAttention(D_GF),
        nn.ConvTranspose2d(D_GF, D_GF, 4, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(D_GF, D_GF, 4, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
    )


class GeneratorN(nn.Module):
    def __init__(self, use_self_attention=False):
        super().__init__()
        self.residuals = nn.Sequential(*[Residual(D_GF * 2) for _ in range(RESIDUALS)])
        self.attn = Attention(D_GF, D_HIDDEN)
        self.upsample = upsample_block(D_GF * 2, D_GF)
        self.use_self_attention = use_self_attention

        if self.use_self_attention:
            self.self_attn = self_attn_block()

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
        # Image-text attention first, image-image attention second
        if self.use_self_attention:
            c_code = self.self_attn(c_code)

        out_code = torch.cat((h_code, c_code), 1)
        out_code = self.residuals(out_code)
        out_code = self.upsample(out_code)  # D_GF/2 x 2ih x 2iw

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
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        self.cond_aug = CondAug(self.device)
        self.f0 = Generator0().to(self.device)
        self.f1 = GeneratorN().to(self.device)
        self.f2 = GeneratorN().to(self.device)
        self.img0 = ImageGen().to(self.device)
        self.img1 = ImageGen().to(self.device)
        self.img2 = ImageGen().to(self.device)

    def to(self, device):
        self.device = device
        super().to(self.device)
        return self

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
        self.fc = nn.Linear(D_HIDDEN, D_COND * 4, bias=True).to(self.device)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

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


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.q_proj = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, c, h, w = x.shape
        q = self.q_proj(x).view(batch, -1, w * h).permute(0, 2, 1)  # Batch x W*H x C(proj)
        k = self.k_proj(x).view(batch, -1, w * h)  # Batch x C(proj) x W*H
        v = self.v_proj(x).view(batch, -1, w * h)  # Batch x C(proj) x W*H

        score = q @ k  # Batch x W*H x W*H
        score = F.softmax(score, -1)

        out = v @ score.permute(0, 2, 1)  # Batch x C x W*H
        out = out.view(batch, c, w, h)  # Batch x C x W x H
        out = self.gamma * out + x
        return out
