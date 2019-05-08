import torch
from torch import nn
import numpy as np

from src.config import GAMMA_3
from src.encoder import ImageEncoder, TextEncoder


def cos_sim(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class DAMSM:
    def __init__(self):
        pass

    def train(self, data, labels):
        img_enc = ImageEncoder()
        txt_enc = TextEncoder(vocab_size=len(data.vocab))
        word_loss1 = 0
        word_loss2 = 0
        sent_loss1 = 0
        sent_loss2 = 0

    def sentence_loss(self, img_code, sent_code, cls_labels, origin_labels, eps=1e-8):
        # in: img_code, sent_code -> BATCH x D_HIDDEN
        masks = []
        cls_labels = np.asarray(cls_labels)
        for i, l in enumerate(cls_labels):
            mask = cls_labels == l
            mask[i] = 0
            masks.append(mask)
        masks = torch.ByteTensor(masks)  # TODO cuda!

        # -> 1 x BATCH X D_HIDDEN
        img_code = img_code.unsqueeze(0)
        sent_code = sent_code.unsqueeze(0)

        # -> 1 x BATCH x 1
        img_norm = torch.norm(img_code, 2, dim=2, keepdim=True)
        sent_norm = torch.norm(sent_code, 2, dim=2, keepdim=True)

        scores1 = torch.bmm(img_code, sent_code.transpose(1, 2))  # -> 1 x BATCH x BATCH
        norm = torch.bmm(img_norm, sent_norm.transpose(1, 2))  # -> 1 x BATCH x BATCH
        scores1 = scores1 / norm.clamp(min=eps) * GAMMA_3  # -> 1 x BATCH x BATCH

        scores1 = scores1.squeeze()  # -> BATCH x BATCH
        scores1.data.masked_fill_(masks, -float('inf'))

        scores2 = scores1.transpose(0, 1)

        # CrossEntropyLoss has builtin softmax
        loss1 = nn.CrossEntropyLoss()(scores1, origin_labels)
        loss2 = nn.CrossEntropyLoss()(scores2, origin_labels)

        return loss1, loss2
