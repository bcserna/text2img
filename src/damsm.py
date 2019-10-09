import json

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.attention import func_attention
from src.config import GAMMA_3, BATCH, GAMMA_1, CAP_MAX_LEN, GAMMA_2, DEVICE, DAMSM_LR, MODEL_DIR
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
    masks = torch.BoolTensor(masks)
    return masks


class DAMSM:
    def __init__(self, vocab_size, device=DEVICE):
        self.device = device
        self.img_enc = ImageEncoder(device=self.device)
        self.txt_enc = TextEncoder(vocab_size=vocab_size, device=self.device)

    def to(self, device):
        self.device = device
        self.img_enc.to(self.device)
        self.txt_enc.to(self.device)
        return self

    def train(self, dataset, epoch, batch_size=BATCH, patience=20):
        loader_config = {
            'batch_size': batch_size,
            'shuffle': True,
            'drop_last': True,
            'collate_fn': dataset.collate_fn
        }
        train_loader = DataLoader(dataset.train, **loader_config)
        test_loader = DataLoader(dataset.test, **loader_config)

        params = list(self.txt_enc.parameters())
        for v in self.img_enc.parameters():
            if v.requires_grad:
                params.append(v)

        optim = torch.optim.Adam(params, lr=DAMSM_LR, betas=(0.5, 0.999))

        img_cap_pair_labels = nn.Parameter(torch.LongTensor(range(batch_size)), requires_grad=False).to(self.device)

        losses = {'train': [], 'test': []}
        patience_step = 0
        min_test_loss = float('Inf')
        min_test_loss_epoch = 0
        for e in tqdm(range(epoch), desc='Epochs', leave=True, dynamic_ncols=True):
            self.img_enc.train(), self.txt_enc.train()
            avg_train_loss = 0
            avg_test_loss = 0

            train_pbar = tqdm(train_loader, leave=False, desc='Training', dynamic_ncols=True)
            for step, batch in enumerate(train_pbar):
                self.img_enc.zero_grad(), self.txt_enc.zero_grad()

                loss, w1_loss, w2_loss, s1_loss, s2_loss = self.batch_loss(batch, img_cap_pair_labels)
                train_pbar.set_description(
                    f'Training (total: {loss.item() / batch_size:05.4f}  '
                    f'w1: {w1_loss / batch_size:05.4f}  '
                    f'w2: {w2_loss / batch_size:05.4f}  '
                    f's1: {s1_loss / batch_size:05.4f}  '
                    f's2: {s2_loss / batch_size:05.4f})')

                avg_train_loss += loss.item()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.txt_enc.parameters(), 0.25)
                optim.step()

            self.img_enc.eval(), self.txt_enc.eval()
            with torch.no_grad():
                for i, b in enumerate(tqdm(test_loader, leave=True, desc='Evaluating test set', dynamic_ncols=True)):
                    loss = self.batch_loss(b, img_cap_pair_labels)[0]
                    avg_test_loss += loss

                avg_train_loss /= (len(train_loader) * batch_size)
                avg_test_loss /= (len(test_loader) * batch_size)
                losses['train'].append(avg_train_loss)
                losses['test'].append(avg_test_loss)

            if avg_test_loss < min_test_loss:
                new_best = '!'
                self.save(f'epoch_{e}')
                self.remove_previous_best(f'epoch_{min_test_loss}')
                min_test_loss = avg_test_loss
                min_test_loss_epoch = e
                patience_step = 0
            else:
                new_best = ''
                patience_step += 1

            sep = '_' * 10
            tqdm.write(f'{sep}Epoch {e}{sep}')
            tqdm.write(f'Avg train loss: {avg_train_loss:05.4f}')
            tqdm.write(f'Avg test loss: {avg_test_loss:05.4f} {new_best}')

            if patience_step == patience:
                tqdm.write(f'Early stopping at epoch {e}')
                tqdm.write(
                    f'Loading model at epoch {min_test_loss_epoch} with best test avg loss {min_test_loss:05.4f}')
                self.load_(f'epoch_{min_test_loss_epoch}')
                return losses

        return losses

    def batch_loss(self, batch, img_cap_pair_labels):
        img_local, img_global = self.img_enc(batch['img256'])
        word_emb, sent_emb = self.txt_enc(batch['caption'])

        w1_loss, w2_loss, _ = self.words_loss(img_local, word_emb, batch['label'], img_cap_pair_labels)
        s1_loss, s2_loss = self.sentence_loss(img_global, sent_emb, batch['label'], img_cap_pair_labels)

        loss = w1_loss + w2_loss + s1_loss + s2_loss
        return loss, w1_loss, w2_loss, s1_loss, s2_loss

    def save(self, name, save_dir=MODEL_DIR):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.txt_enc.state_dict(), f'{save_dir}/{name}_text_enc.pt')
        torch.save(self.img_enc.state_dict(), f'{save_dir}/{name}_img_enc.pt')
        config = {'vocab_size': self.txt_enc.vocab_size}
        with open(f'{save_dir}/{name}_config.json', 'w') as f:
            json.dump(config, f)

    @staticmethod
    def remove_previous_best(name, save_dir=MODEL_DIR):
        os.remove(f'{save_dir}/{name}_text_enc.pt')
        os.remove(f'{save_dir}/{name}_img_enc.pt')
        os.remove(f'{save_dir}/{name}_config.json')

    @staticmethod
    def load(name, load_dir=MODEL_DIR):
        with open(f'{load_dir}/{name}_config.json', 'r') as f:
            config = json.load(f)
        damsm = DAMSM(config['vocab_size'])
        damsm.load_(name)
        return damsm

    def load_(self, name, load_dir=MODEL_DIR):
        self.txt_enc.load_state_dict(torch.load(f'{load_dir}/{name}_text_enc.pt'))
        self.img_enc.load_state_dict(torch.load(f'{load_dir}/{name}_img_enc.pt'))
        self.txt_enc.eval(), self.img_enc.eval()

    def sentence_loss(self, img_code, sent_code, cls_labels, img_cap_pair_label, eps=1e-8):
        # in: img_code, sent_code -> BATCH x D_HIDDEN
        # mask samples belonging to the same class
        masks = get_class_masks(cls_labels).to(self.device)

        # -> BATCH x 1
        img_norm = img_code.norm(p=2, dim=1, keepdim=True)
        sent_norm = sent_code.norm(p=2, dim=1, keepdim=True)

        scores1 = img_code @ sent_code.transpose(0, 1)  # -> BATCH x BATCH
        norm = img_norm @ sent_norm.transpose(0, 1)  # -> BATCH x BATCH
        scores1 = scores1 / norm.clamp(min=eps) * GAMMA_3  # -> BATCH x BATCH

        scores1.data.masked_fill_(masks, -float('inf'))
        scores2 = scores1.transpose(0, 1)

        # nn.CrossEntropyLoss has builtin softmax
        loss1 = F.cross_entropy(scores1, img_cap_pair_label)
        loss2 = F.cross_entropy(scores2, img_cap_pair_label)

        return loss1, loss2

    def words_loss(self, img_features, word_embs, cls_labels, img_cap_pair_label):
        # img_features (local features of the image): BATCH x D_HIDDEN x 17 x 17
        # word_embs: BATCH x D_HIDDEN x CAP_LEN

        masks = get_class_masks(cls_labels)
        att_maps = []
        similarities = []

        batch_size = img_features.size(0)

        for i in range(batch_size):
            words = word_embs[i].unsqueeze(0).contiguous()  # -> 1 x D_HIDDEN x CAP_LEN
            words = words.repeat(batch_size, 1, 1)  # -> BATCH x D_HIDDEN x CAP_LEN
            # region_context: word representation by image regions
            region_context, att_map = func_attention(words, img_features, GAMMA_1)
            att_maps.append(att_map[i].unsqueeze(0).contiguous())
            # BATCH * CAP_LEN x D_HIDDEN
            words = words.transpose(1, 2).contiguous().view(batch_size * CAP_MAX_LEN, -1)
            region_context = region_context.transpose(1, 2).contiguous().view(batch_size * CAP_MAX_LEN, -1)

            # Eq. (10)
            sim = cos_sim(words, region_context).view(batch_size, CAP_MAX_LEN)
            sim.mul_(GAMMA_2).exp_()
            sim = sim.sum(dim=1, keepdim=True)
            sim = torch.log(sim)
            # similarities(i, j): the similarity between the i-th image and the j-th text description
            similarities.append(sim)

        similarities = torch.cat(similarities, 1)  # -> BATCH x BATCH
        masks = masks.view(batch_size, batch_size).contiguous().to(self.device)

        similarities = similarities * GAMMA_3
        similarities.data.masked_fill_(masks, -float('inf'))

        similarities2 = similarities.transpose(0, 1)
        loss1 = nn.CrossEntropyLoss()(similarities, img_cap_pair_label)
        loss2 = nn.CrossEntropyLoss()(similarities2, img_cap_pair_label)

        return loss1, loss2, att_maps
