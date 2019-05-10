import torch
from torch import nn
import numpy as np

from src.attention import func_attention
from src.config import GAMMA_3, CUDA, BATCH, GAMMA_1, CAP_LEN, GAMMA_2
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
        # img_features (local features of the image): BATCH x D_HIDDEN x 17 x 17
        # word_embs: BATCH x D_HIDDEN x CAP_LEN

        masks = get_class_masks(cls_labels)
        att_maps = []
        similarities = []

        for i in range(BATCH):
            words = word_embs[i].unsqueeze(0).contiguous()  # -> 1 x D_HIDDEN x CAP_LEN
            words = words.repeat(BATCH, 1, 1)  # -> BATCH x D_HIDDEN x CAP_LEN
            # region_context: word representation by image regions
            region_context, att_map = func_attention(words, img_features, GAMMA_1)
            att_maps.append(att_map[i].unsqueeze(0).contiguous())
            # BATCH * CAP_LEN x D_HIDDEN
            words = words.transpose(1, 2).contiguous().view(BATCH * CAP_LEN, -1)
            region_context = region_context.transpose(1, 2).contiguous().view(BATCH * CAP_LEN, -1)

            # Eq. (10)
            sim = cos_sim(words, region_context).view(BATCH, CAP_LEN)
            sim.mul_(GAMMA_2).exp_()
            sim = sim.sum(dim=1, keepdim=True)
            sim = torch.log(sim)
            # similarities(i, j): the similarity between the i-th image and the j-th text description
            similarities.append(sim)

        similarities = torch.cat(similarities, 1)  # -> BATCH x BATCH
        masks = masks.view(BATCH, BATCH).contiguous()

        similarities = similarities * GAMMA_3
        similarities.data.masked_fill_(masks, -float('inf'))

        similarities2 = similarities.transpose(0, 1)
        loss1 = nn.CrossEntropyLoss()(similarities, origin_labels)
        loss2 = nn.CrossEntropyLoss()(similarities2, origin_labels)

        return loss1, loss2, att_maps
