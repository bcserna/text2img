import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.config import D_HIDDEN, P_DROP, D_WORD, BATCH


class TextEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=D_WORD)
        self.emb_dropout = nn.Dropout(P_DROP)
        self.rnn = nn.LSTM(
            input_size=D_WORD,
            hidden_size=D_HIDDEN // 2,  # bidirectional
            batch_first=True,
            # dropout=P_DROP,
            bidirectional=True)
        # Initial cell and hidden state for each sequence
        self.hidden0 = nn.Parameter(torch.randn(D_HIDDEN // 2), requires_grad=True)
        self.cell0 = nn.Parameter(torch.randn(D_HIDDEN // 2), requires_grad=True)

    def forward(self, x):
        # sort = np.argsort(cap_lens)[::-1]
        # cap_lens = np.asarray(cap_lens)[sort]
        # x = np.asarray(x)[sort]

        e = self.embed(torch.tensor(x, dtype=torch.int64))
        e = self.emb_dropout(e)
        # e = pack_padded_sequence(e, cap_lens, batch_first=True)
        out, hidden = self.rnn(e, (self.hidden0.repeat(2, BATCH, 1), self.cell0.repeat(2, BATCH, 1)))
        # words_repr = pad_packed_sequence(out, batch_first=True)[0]
        # words_repr = out.transpose(1, 2)
        return out, hidden


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception_model = torchvision.models.inception_v3(pretrained=True)
        # Freeze Inception V3 parameters
        for param in self.inception_model.parameters():
            param.requires_grad = False
        # 768: the dimension of mixed_6e layer's sub-regions (768 x 289 [number of sub-regions])
        self.local_proj = nn.Linear(768, D_HIDDEN)
        # 2048: the dimension of last average pool's output
        self.global_proj = nn.Linear(2048, D_HIDDEN)

        # self.local_proj.weight.data.uniform_(-IMG_WEIGHT_INIT_RANGE, IMG_WEIGHT_INIT_RANGE)
        # self.global_proj.weight.data.uniform_(-IMG_WEIGHT_INIT_RANGE, IMG_WEIGHT_INIT_RANGE)

    def forward(self, x):
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.inception_model.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.inception_model.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.inception_model.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.inception_model.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.inception_model.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.inception_model.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.inception_model.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.inception_model.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.inception_model.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.inception_model.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.inception_model.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.inception_model.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.inception_model.Mixed_6e(x)
        # 17 x 17 x 768

        local_features = x.view(-1, 289, 768)
        # 289 x 768

        # 17 x 17 x 768
        x = self.inception_model.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.inception_model.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.inception_model.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        local_features = self.local_proj(local_features)
        global_features = self.global_proj(x)

        return local_features, global_features


class CondAug(nn.Module):
    def __init__(self):
        super().__init__()

