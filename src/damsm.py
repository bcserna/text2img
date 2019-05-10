import torch
from torch import nn
import numpy as np

from src.config import GAMMA_3, CUDA, BATCH
from src.encoder import ImageEncoder, TextEncoder


def cos_sim(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def get_class_masks(cls_labels):
    masks = []
    cls_labels = np.asarray(cls_labels)
    for i, l in enumerate(cls_labels):
        mask = cls_labels == l
        mask[i] = 0
        masks.append(mask.reshape((1, -1)))
    masks = torch.ByteTensor(masks)
    if CUDA:
        masks = masks.cuda()

    return masks


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

    @staticmethod
    def sentence_loss(img_code, sent_code, cls_labels, origin_labels, eps=1e-8):
        # in: img_code, sent_code -> BATCH x D_HIDDEN
        # mask samples belonging to the same class
        masks = get_class_masks(cls_labels)

        # -> BATCH x 1
        img_norm = img_code.norm(p=2, dim=1, keepdim=True)
        sent_norm = sent_code.norm(p=2, dim=1, keepdim=True)

        scores1 = img_code @ sent_code.transpose(0, 1)  # -> BATCH x BATCH
        norm = img_norm @ sent_norm.transpose(0, 1)  # -> BATCH x BATCH
        scores1 = scores1 / norm.clamp(min=eps) * GAMMA_3  # -> BATCH x BATCH

        scores1.data.masked_fill_(masks, -float('inf'))
        scores2 = scores1.transpose(0, 1)

        # nn.CrossEntropyLoss has builtin softmax
        loss1 = nn.CrossEntropyLoss()(scores1, origin_labels)
        loss2 = nn.CrossEntropyLoss()(scores2, origin_labels)

        return loss1, loss2

    @staticmethod
    def words_loss(img_features, word_embs, cls_labels, origin_labels):
        # img_features: BATCH x D_HIDDEN x 17 x 17
        # word_embs: BATCH x D_HIDDEN x cap_len
        masks = get_class_masks(cls_labels)

        for i in range(BATCH):

