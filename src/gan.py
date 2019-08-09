import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import *
from src.discriminator import Discriminator
from src.generator import Generator
from src.util import roll_tensor


class AttnGAN(object):
    def __init__(self, damsm, device=DEVICE):
        self.gen = Generator().to(device)
        self.disc = Discriminator().to(device)
        self.damsm = damsm
        self.device = device

    def train(self, dataset, epoch, batch=BATCH):
        loader_config = {
            'batch_size': batch,
            'shuffle': True,
            'drop_last': True,
            'collate_fn': dataset.collate_fn
        }
        train_loader = DataLoader(dataset.train, **loader_config)
        # test_loader = DataLoader(dataset.test, **loader_config)

        losses = {}

        gen_optimizer = torch.optim.Adam(self.gen.parameters(),
                                         lr=GENERATOR_LR,
                                         betas=(0.5, 0.999))

        discriminators = [self.disc.d64, self.disc.d128, self.disc.d256]
        disc_optimizers = [torch.optim.Adam(d.parameters(),
                                            lr=DISCRIMINATOR_LR,
                                            betas=(0.5, 0.999))
                           for d in discriminators]

        real_labels = nn.Parameter(torch.FloatTensor(batch).fill_(1), requires_grad=False).to(self.device)
        fake_labels = nn.Parameter(torch.FloatTensor(batch).fill_(0), requires_grad=False).to(self.device)
        match_labels = nn.Parameter(torch.LongTensor(range(batch)), requires_grad=False).to(self.device)

        noise = nn.Parameter(torch.FloatTensor(batch, D_Z), requires_grad=False).to(self.device)
        for e in tqdm(range(epoch), desc='Epochs'):
            self.gen.train(), self.disc.train()

            train_pbar = tqdm(train_loader, desc='Training', leave=False)
            for batch in train_pbar:
                self.gen.zero_grad(), self.disc.zero_grad()
                word_embs, sent_embs = self.damsm.txt_enc(batch['caption'])
                word_embs, sent_embs = word_embs.detach(), sent_embs.detach()
                attn_mask = torch.tensor(batch['caption']).to(self.device) == dataset.vocab[END_TOKEN]
                # Generate images
                noise.data.normal_(0, 1)
                generated, att, mu, logvar = self.gen(noise, sent_embs, word_embs, attn_mask)
                # Discriminator loss
                real_imgs = [batch['img64'], batch['img128'], batch['img256']]
                real_features = self.disc(real_imgs)
                fake_features = self.disc(generated)

                real_logits = self.disc.get_logits(real_features, sent_embs)
                fake_logits = self.disc.get_logits(fake_features, sent_embs)

                real_errors = [nn.functional.binary_cross_entropy(l, real_labels) for l in real_logits]
                # Contributes to generator loss too
                fake_errors = [nn.functional.binary_cross_entropy(l, fake_labels) for l in fake_logits]

                mismatched_logits = self.disc.get_logits(real_features, roll_tensor(sent_embs, 1))
                mismatched_errors = [nn.functional.binary_cross_entropy(l, fake_labels) for l in mismatched_logits]

                disc_errors = [real + fake + mismatched for real, fake, mismatched in
                               zip(real_errors, fake_errors, mismatched_errors)]

                for error, optimizer in zip(disc_errors, disc_optimizers):
                    error.backward(retain_graph=True)
                    optimizer.step()

                # Generator loss
                local_features, global_features = self.damsm.img_enc(generated[-1])

                w1_loss, w2_loss, _ = self.damsm.words_loss(local_features, word_embs, batch['label'], match_labels)
                w_loss = (w1_loss + w2_loss) * LAMBDA

                s1_loss, s2_loss = self.damsm.sentence_loss(global_features, sent_embs, batch['label'], match_labels)
                s_loss = (s1_loss + s2_loss) * LAMBDA
                s_loss.backward(retain_graph=True)

                kl_loss = self.KL_loss(mu, logvar)
                kl_loss.backward()

                gen_optimizer.step()

                g_total = w_loss + s_loss + kl_loss
                for error in fake_errors:
                    g_total += error

                train_pbar.set_description(f'Training (G_total: {g_total:05.3f})')
                # TODO log more metrics
                # TODO averaging generator params?

    @staticmethod
    def KL_loss(mu, logvar):
        loss = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        loss = torch.mean(loss).mul_(-0.5)
        return loss
