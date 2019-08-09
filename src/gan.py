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

    def train(self, dataset, epoch, batch_size=GAN_BATCH):
        loader_config = {
            'batch_size': batch_size,
            'shuffle': True,
            'drop_last': True,
            'collate_fn': dataset.collate_fn
        }
        train_loader = DataLoader(dataset.train, **loader_config)
        # test_loader = DataLoader(dataset.test, **loader_config)

        losses = {'g': [], 'd': []}

        gen_optimizer = torch.optim.Adam(self.gen.parameters(),
                                         lr=GENERATOR_LR,
                                         betas=(0.5, 0.999))

        discriminators = [self.disc.d64, self.disc.d128, self.disc.d256]
        disc_optimizers = [torch.optim.Adam(d.parameters(),
                                            lr=DISCRIMINATOR_LR,
                                            betas=(0.5, 0.999))
                           for d in discriminators]

        real_labels = nn.Parameter(torch.FloatTensor(batch_size).fill_(1), requires_grad=False).to(self.device)
        fake_labels = nn.Parameter(torch.FloatTensor(batch_size).fill_(0), requires_grad=False).to(self.device)
        match_labels = nn.Parameter(torch.LongTensor(range(batch_size)), requires_grad=False).to(self.device)

        noise = nn.Parameter(torch.FloatTensor(batch_size, D_Z), requires_grad=False).to(self.device)
        for e in tqdm(range(epoch), desc='Epochs'):
            self.gen.train(), self.disc.train()
            g_loss = 0
            d_loss = [0, 0, 0]

            train_pbar = tqdm(train_loader, desc='Training', leave=False)
            for batch in train_pbar:
                self.gen.zero_grad(), self.disc.zero_grad()
                with torch.no_grad():
                    word_embs, sent_embs = self.damsm.txt_enc(batch['caption'])
                # word_embs, sent_embs = word_embs.detach(), sent_embs.detach()
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

                for error, optimizer, disc in zip(disc_errors, disc_optimizers, range(3)):
                    error.backward(retain_graph=True)
                    optimizer.step()
                    d_loss[disc] += error.item() / batch_size

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

                avg_g_loss = g_total.item() / batch_size
                g_loss += avg_g_loss

                train_pbar.set_description(f'Training (G: {avg_g_loss:05.4f}'
                                           f'  D64: {disc_errors[0] / batch_size:05.4f}'
                                           f'  D128: {disc_errors[1] / batch_size:05.4f}'
                                           f'  D256: {disc_errors[2] / batch_size:05.4f})')

            g_loss /= len(train_loader)
            for i, _ in enumerate(d_loss):
                d_loss[i] /= len(train_loader)

            losses['g'].append(g_loss)
            losses['d'].append(d_loss)

            sep = '_'
            tqdm.write(f'{sep}Epoch {e}{sep}')
            tqdm.write(f'Avg generator loss: {g_loss:05.4f}')
            tqdm.write(f'Avg discriminator loss: 64 {d_loss[0]:05.4f}  128 {d_loss[1]:05.4f}  256 {d_loss[1]:05.4f}')

        return losses

    @staticmethod
    def KL_loss(mu, logvar):
        loss = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        loss = torch.mean(loss).mul_(-0.5)
        return loss
