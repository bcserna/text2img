import torch
from torch import nn
import numpy as np
import time
import os
import pickle
from tqdm import tqdm

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
        masks.append(mask)
    masks = torch.ByteTensor(masks)
    if CUDA:
        masks = masks.cuda()

    return masks


class DAMSM:
    def __init__(self, vocab_size):
        self.img_enc = ImageEncoder()
        self.txt_enc = TextEncoder(vocab_size=vocab_size)

    def train(self):
        self.img_enc.train(), self.txt_enc.train()

        params = list(self.txt_enc.parameters())
        for v in self.img_enc.parameters():
            if v.requires_grad:
                params.append(v)
        optim = torch.optim.Adam(params, lr=2e-4, betas=(0.5, 0.999))
        word_loss1 = 0
        word_loss2 = 0
        sent_loss1 = 0
        sent_loss2 = 0

        start_time = time.time()
        img_cap_pair_labels = nn.Parameter(torch.LongTensor(range(BATCH)), requires_grad=False)
        for step, batch in enumerate(tqdm(self.data.loader)):
            self.img_enc.zero_grad()
            self.txt_enc.zero_grad()

            img_local, img_global = self.img_enc(batch['img256'])
            word_emb, sent_emb = self.txt_enc(batch['caption'])

            w1_loss, w2_loss, _ = self.words_loss(img_local, word_emb, batch['label'], img_cap_pair_labels)
            s1_loss, s2_loss = self.sentence_loss(img_global, sent_emb, batch['label'], img_cap_pair_labels)

            loss = w1_loss + w2_loss + s1_loss + s2_loss

            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm(self.txt_enc.parameters(), 0.25)
            optim.step()

            tqdm.write(f'\nw1: {w1_loss}    w2: {w2_loss}    s1: {s1_loss}    s2: {s2_loss}    total: {loss}')

    def save(self, name):
        save_dir = 'models'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.txt_enc.state_dict(), f'{save_dir}/{name}_text_enc.pt')
        torch.save(self.img_enc.state_dict(), f'{save_dir}/{name}_img_enc.pt')
        config = {'vocab_size': self.txt_enc.vocab_size}
        with open(f'{save_dir}/{name}_config.pkl', 'wb') as f:
            pickle.dump(config, f)

    @staticmethod
    def load(name):
        load_dir = 'models'
        with open(f'{load_dir}/{name}_config.pkl', 'rb') as f:
            config = pickle.load(f)
        damsm = DAMSM(config['vocab_size'])
        damsm.txt_enc.load_state_dict(torch.load(f'{load_dir}/{name}_text_enc.pt'))
        damsm.img_enc.load_state_dict(torch.load(f'{load_dir}/{name}_img_enc.pt'))
        return damsm

    @staticmethod
    def sentence_loss(img_code, sent_code, cls_labels, img_cap_pair_label, eps=1e-8):
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
        loss1 = nn.CrossEntropyLoss()(scores1, img_cap_pair_label)
        loss2 = nn.CrossEntropyLoss()(scores2, img_cap_pair_label)

        return loss1, loss2

    @staticmethod
    def words_loss(img_features, word_embs, cls_labels, img_cap_pair_label):
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
        loss1 = nn.CrossEntropyLoss()(similarities, img_cap_pair_label)
        loss2 = nn.CrossEntropyLoss()(similarities2, img_cap_pair_label)

        return loss1, loss2, att_maps
