import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import math
from scipy.stats import entropy
from tqdm import tqdm

from src.config import GAN_BATCH, DEVICE, D_Z, END_TOKEN


def inception_score(gan, dataset, inception_model, batch_size=GAN_BATCH, samples=50000, splits=10, device=DEVICE):
    with torch.no_grad():
        inception_preds = np.zeros((samples, 1000))

        loader = DataLoader(dataset.test, batch_size=batch_size, shuffle=True, drop_last=True,
                            collate_fn=dataset.collate_fn)

        epochs = math.ceil(samples / (len(loader) * batch_size))
        nb_generated = 0
        for _ in tqdm(range(epochs), desc='Generating samples for inception score'):
            for batch in loader:
                word_embs, sent_embs = gan.damsm.txt_enc(batch['caption'])
                attn_mask = torch.tensor(batch['caption']).to(device) == dataset.vocab[END_TOKEN]

                # Generate images
                noise = torch.FloatTensor(batch_size, D_Z).to(device)
                noise.data.normal_(0, 1)
                generated, att, mu, logvar = gan.gen(noise, sent_embs, word_embs, attn_mask)
                x = generated[-1]
                x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
                x = inception_model(x)
                x = F.softmax(x, dim=-1).data.cpu().numpy()

                samples_left = samples - nb_generated
                if samples_left < batch_size:
                    inception_preds[nb_generated:] = x[:samples_left]
                    break
                else:
                    inception_preds[nb_generated:nb_generated + batch_size] = x
                    nb_generated += batch_size

        scores = []
        split_size = samples // splits
        for s in range(splits):
            split_scores = []

            split = inception_preds[s * split_size: (s + 1) * split_size]
            p_y = np.mean(split, axis=0)
            for sample_pred in split:
                split_scores.append(entropy(sample_pred, p_y))

            scores.append(np.exp(np.mean(split_scores)))

        return np.mean(scores), np.std(scores)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def rotate_tensor(t, n, dim=0):
    return torch.cat((t[-n:], t[:-n]), dim=dim)


def count_params(module):
    trainable = 0
    non_trainable = 0
    for p in module.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            non_trainable += p.numel()
    return trainable, non_trainable


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.factor, mode=self.mode)


def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)


def upsample_block(in_planes, out_planes):
    return nn.Sequential(
        Interpolate(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        nn.modules.activation.GLU(dim=1)
    )


def residual_block(channels):
    return nn.Sequential(
        conv3x3(channels, channels * 2),
        nn.BatchNorm2d(channels * 2),
        nn.modules.activation.GLU(dim=1),
        conv3x3(channels, channels),
        nn.BatchNorm2d(channels)
    )
