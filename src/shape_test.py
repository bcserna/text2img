import torch
from torch import nn
from torch.autograd import Variable

from src.config import BATCH, D_Z, D_HIDDEN, D_COND, D_GF, D_WORD, CAP_MAX_LEN
from src.damsm import DAMSM
from src.data import CUB
from src.discriminator import Discriminator64, Discriminator128, Discriminator256
from src.encoder import TextEncoder, ImageEncoder
from src.generator import Generator0, GeneratorN, ImageGen, Generator


def test_run_encoders():
    data = CUB()
    text_encoder = TextEncoder(vocab_size=len(data.vocab))
    img_encoder = ImageEncoder()
    text_samples = []
    img_samples = []
    for i in range(BATCH):
        text_samples.append(data[i][1])
        img_samples.append(data[i][0])

    print('image sample shapes:')
    for i in img_samples[0]:
        print(i.size())

    print(f'text sample shape: {len(text_samples[0])}')

    txt_out, txt_hidden = text_encoder(text_samples)
    high_res_imgs = torch.stack([i[2] for i in img_samples])
    img_local, img_global = img_encoder(high_res_imgs)
    print(f'encoded image local features\' shape: {img_local.size()}    global features\' shape: {img_global.size()}')
    print(f'word emb shape shape: {txt_out.size()}    sentence emb shape: {txt_hidden.size()}')
    return txt_out, txt_hidden, img_local, img_global


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


def test_run_image_gen():
    g = ImageGen()
    h_code64 = torch.randn(BATCH, D_GF, 64, 64)
    h_code128 = torch.randn(BATCH, D_GF, 128, 128)
    h_code256 = torch.randn(BATCH, D_GF, 256, 256)
    out64 = g(h_code64)
    out128 = g(h_code128)
    out256 = g(h_code256)
    print(f'out64: {out64.size()}    out128: {out128.size()}    out256: {out256.size()}')
    return out64, out128, out256


def test_run_generator():
    g = Generator()
    z_code = torch.randn(BATCH, D_Z)
    sent_emb = torch.randn(BATCH, D_HIDDEN)
    word_embs = torch.randn(BATCH, D_WORD, CAP_MAX_LEN)
    gen, att, mu, logvar = g(z_code, sent_emb, word_embs, None)
    print('Generated shape:')
    for k in gen:
        print(gen[k].size())
    print('Attention shape:')
    for a in att:
        print(att[a].size())

    print(f'Mu shape: {mu.size()}    logvar shape: {logvar.size()}')
    return gen, att, mu, logvar


def test_run_discriminator128():
    d = Discriminator128()
    x = torch.randn(BATCH, 3, 128, 128)
    o = d(x)
    print(f'Discriminator128 output shape: {o.size()}')
    return o


def test_run_discriminator256():
    d = Discriminator256()
    x = torch.randn(BATCH, 3, 256, 256)
    o = d(x)
    print(f'Discriminator256 output shape: {o.size()}')
    return o


def test_run_damsm_losses():
    data = CUB()
    text_enc = TextEncoder(len(data.vocab))
    img_enc = ImageEncoder()
    for batch in data.loader:
        img64 = batch['img64']
        img128 = batch['img128']
        img256 = batch['img256']
        cap = batch['caption']
        label = batch['label']
        print(
            f'img64 shape: {img64.size()}',
            f'img128 shape: {img128.size()}',
            f'img256 shape: {img256.size()}',
            f'caption shape: {len(cap)} x {len(cap[0])}',
            f'label shape: {len(label)}',
            sep='\n'
        )

        img_code = img_enc(img256)
        print(f'local image features: {img_code[0].size()}    global image features: {img_code[1].size()}')
        text_code = text_enc(cap)
        print(f'word embeddings: {text_code[0].size()}    sentence embedding: {text_code[1].size()}')

        img_cap_pair_label = nn.Parameter(torch.LongTensor(range(BATCH)), requires_grad=False)
        sent_loss = DAMSM.sentence_loss(img_code[1], text_code[1], label, img_cap_pair_label)
        print(f'sentence loss 1: {sent_loss[0]}    sentence loss 2: {sent_loss[1]}')
        words_loss = DAMSM.words_loss(img_code[0], text_code[0], label, img_cap_pair_label)
        print(f'words loss 1: {words_loss[0]}    words loss 2: {words_loss[1]}')
        return
