import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import warnings

from src.config import D_HIDDEN, P_DROP, D_WORD, IMG_WEIGHT_INIT_RANGE, DEVICE
from src.util import conv1x1, count_params


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, device=DEVICE):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=D_WORD).to(self.device)
        self.emb_dropout = nn.Dropout(P_DROP).to(self.device)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.rnn = nn.LSTM(
                input_size=D_WORD,
                hidden_size=D_HIDDEN // 2,  # bidirectional
                batch_first=True,
                dropout=P_DROP,
                bidirectional=True).to(self.device)
        # Initial cell and hidden state for each sequence
        self.hidden0 = nn.Parameter(torch.randn(D_HIDDEN // 2), requires_grad=True).to(self.device)
        self.cell0 = nn.Parameter(torch.randn(D_HIDDEN // 2), requires_grad=True).to(self.device)
        # Different initial hidden state for different directions?

        p_trainable, p_non_trainable = count_params(self)
        print(f'Text encoder params: trainable {p_trainable} - non_trainable {p_non_trainable}')

    def forward(self, x):
        batch_size = len(x)
        init_hidden = self.hidden0.repeat(2, batch_size, 1)
        init_cell = self.cell0.repeat(2, batch_size, 1)

        e = self.embed(torch.tensor(x, dtype=torch.int64).to(self.device))
        e = self.emb_dropout(e)
        word_embs, hidden = self.rnn(e, (init_hidden, init_cell))
        word_embs = word_embs.transpose(1, 2)  # -> BATCH x D_HIDDEN x seq_len
        sent_embs = hidden[0].transpose(0, 1).contiguous()  # -> BATCH x 2 x D_HIDDEN/2
        sent_embs = sent_embs.view(batch_size, -1)  # -> BATCH x D_HIDDEN
        return word_embs, sent_embs


class ImageEncoder(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        self.inception_model = torchvision.models.inception_v3(pretrained=True).to(self.device)
        # Freeze Inception V3 parameters
        for param in self.inception_model.parameters():
            param.requires_grad = False
        # 768: the dimension of mixed_6e layer's sub-regions (768 x 289 [number of sub-regions, 17 x 17])
        self.local_proj = conv1x1(768, D_HIDDEN).to(self.device)
        # 2048: the dimension of last average pool's output
        self.global_proj = nn.Linear(2048, D_HIDDEN).to(self.device)

        self.local_proj.weight.data.uniform_(-IMG_WEIGHT_INIT_RANGE, IMG_WEIGHT_INIT_RANGE)
        self.global_proj.weight.data.uniform_(-IMG_WEIGHT_INIT_RANGE, IMG_WEIGHT_INIT_RANGE)

        p_trainable, p_non_trainable = count_params(self)
        print(f'Image encoder params: trainable {p_trainable} - non_trainable {p_non_trainable}')

    def forward(self, x):
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.functional.interpolate(input=x, size=(299, 299), mode='bilinear', align_corners=False).to(self.device)
        # x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
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

        local_features = x
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
