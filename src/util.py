from torch import nn
import torch.nn.functional as F

from src.config import BATCH
from src.data import CUB
from src.encoder import TextEncoder, ImageEncoder


def test_run_encoder():
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


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.factor, mode=self.mode)


def upsample(in_planes, out_planes):
    return nn.Sequential(
        Interpolate(scale_factor=2, mode='nearest'),
        # nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.modules.activation.GLU(dim=1)
    )
