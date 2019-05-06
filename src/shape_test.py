import torch

from src.config import BATCH, D_Z, D_HIDDEN, D_COND, D_GF, D_WORD, CAP_MAX_LEN
from src.data import CUB
from src.encoder import TextEncoder, ImageEncoder
from src.generator import Generator0, GeneratorN


def test_run_encoders():
    data = CUB()
    text_encoder = TextEncoder(vocab_size=len(data.vocab))
    img_encoder = ImageEncoder()
    text_samples = []
    img_samples = []
    for i in range(BATCH):
        text_samples.append(data[i][1])
        img_samples.append(data[i][0])

    print(img_samples[0].shape)
    encoded_txt = text_encoder(text_samples)
    encoded_img = img_encoder(img_samples)
    return encoded_txt, encoded_img


def test_run_generator_0():
    g = Generator0()
    z_code = torch.randn(BATCH, D_Z)
    c_code = torch.randn(BATCH, D_COND)
    out = g(z_code, c_code)
    print(f'Generator0 output shape: {out.size()}')
    return out


def test_run_generator_n():
    g1 = GeneratorN()
    g2 = GeneratorN()
    h_code = torch.randn(BATCH, D_GF, 64, 64)
    c_code = torch.randn(BATCH, D_COND)
    word_embs = torch.randn(BATCH, D_WORD, CAP_MAX_LEN)
    mask = None
    out_code, att = g1(h_code, c_code, word_embs, mask)
    print(f'Generator1 output shape: {out_code.size()}    Attention shape: {att.size()}')
    out_code, att = g2(out_code, c_code, word_embs, mask)
    print(f'Generator2 output shape: {out_code.size()}    Attention shape: {att.size()}')
    return out_code, att

