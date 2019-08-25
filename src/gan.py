import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import time
import os
from tqdm import tqdm

from src.config import DEVICE, GAN_BATCH, GENERATOR_LR, DISCRIMINATOR_LR, D_Z, END_TOKEN, LAMBDA
from src.discriminator import Discriminator
from src.generator import Generator
from src.util import roll_tensor


class AttnGAN:
    def __init__(self, damsm, device=DEVICE):
        self.gen = Generator().to(device)
        self.disc = Discriminator().to(device)
        self.damsm = damsm
        self.damsm.txt_enc.eval(), self.damsm.img_enc.eval()
        self.device = device

    def train(self, dataset, epoch, batch_size=GAN_BATCH, test_sample_every=1, nb_test_samples=2):
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        os.makedirs(start_time)

        loader_config = {
            'batch_size': batch_size,
            'shuffle': True,
            'drop_last': True,
            'collate_fn': dataset.collate_fn
        }
        train_loader = DataLoader(dataset.train, **loader_config)

        metrics = {
            'loss': {
                'g': [],
                'd': []
            },
            'accuracy': {
                'real': [],
                'fake': [],
                'mismatched': []
            }
        }

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

        noise = torch.FloatTensor(batch_size, D_Z, requires_grad=False).to(self.device)
        for e in tqdm(range(epoch), desc='Epochs'):
            self.gen.train(), self.disc.train()
            g_loss = 0
            d_loss = [0, 0, 0]
            real_accuracy = [0, 0, 0]
            fake_accuracy = [0, 0, 0]
            mismatched_accuracy = [0, 0, 0]

            train_pbar = tqdm(train_loader, desc='Training', leave=False)
            for batch in train_pbar:
                batch_d_loss = [0, 0, 0]
                batch_real_accuracy = [0, 0, 0]
                batch_fake_accuracy = [0, 0, 0]
                batch_mismatched_accuracy = [0, 0, 0]

                self.gen.zero_grad(), self.disc.zero_grad()
                with torch.no_grad():
                    word_embs, sent_embs = self.damsm.txt_enc(batch['caption'])
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
                    batch_d_loss[disc] = error.item() / batch_size
                    d_loss[disc] += batch_d_loss[disc]

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

                for real, fake, mismatched, i in zip(real_logits, fake_logits, mismatched_logits, range(3)):
                    # Real images should be classified as real
                    batch_real_accuracy[i] = (real > 0.5).sum().item() / real.size(0)
                    # Generated images should be classified as fake
                    batch_fake_accuracy[i] = (fake <= 0.5).sum().item() / fake.size(0)
                    # Images with mismatched descriptions should be classified as fake
                    batch_mismatched_accuracy[i] = (mismatched <= 0.5).sum().item() / mismatched.size(0)

                    real_accuracy[i] += batch_real_accuracy[i]
                    fake_accuracy[i] += batch_fake_accuracy[i]
                    mismatched_accuracy[i] += batch_mismatched_accuracy[i]

                train_pbar.set_description(f'Training (G: {avg_g_loss:05.4f}'
                                           f'  D64: {batch_d_loss[0]:05.4f}'
                                           f'  D128: {batch_d_loss[1]:05.4f}'
                                           f'  D256: {batch_d_loss[2]:05.4f})')

            batches = len(train_loader)
            g_loss /= batches
            for i in range(len(d_loss)):
                d_loss[i] /= batches
                real_accuracy[i] /= batches
                fake_accuracy[i] /= batches
                mismatched_accuracy[i] /= batches

            metrics['loss']['g'].append(g_loss)
            metrics['loss']['d'].append(d_loss)
            metrics['accuracy']['real'].append(real_accuracy)
            metrics['accuracy']['fake'].append(fake_accuracy)
            metrics['accuracy']['mismatched'].append(mismatched_accuracy)

            sep = '_' * 10
            tqdm.write(f'{sep}Epoch {e}{sep}')
            tqdm.write(f'Generator avg loss: {g_loss:05.4f}')
            tqdm.write(f'Discriminator0 avg: '
                       f'loss({d_loss[0]:05.4f})  '
                       f'r-acc({real_accuracy[0]:04.3f})  '
                       f'f-acc({fake_accuracy[0]:04.3f})  '
                       f'm-acc({mismatched_accuracy[0]:04.3f})')
            tqdm.write(f'Discriminator1 avg: '
                       f'loss({d_loss[1]:05.4f})  '
                       f'r-acc({real_accuracy[1]:04.3f})  '
                       f'f-acc({fake_accuracy[1]:04.3f})  '
                       f'm-acc({mismatched_accuracy[1]:04.3f})')
            tqdm.write(f'Discriminator2 avg: '
                       f'loss({d_loss[2]:05.4f})  '
                       f'r-acc({real_accuracy[2]:04.3f})  '
                       f'f-acc({fake_accuracy[2]:04.3f})  '
                       f'm-acc({mismatched_accuracy[2]:04.3f})')

            if e % test_sample_every == 0:
                texts = [dataset.test.data['caption_0'].iloc[i] for i in range(nb_test_samples)]
                generated_samples = self.generate_from_text(texts, dataset)
                self._save_generated(generated_samples, e, start_time)

        return metrics

    @staticmethod
    def KL_loss(mu, logvar):
        loss = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        loss = torch.mean(loss).mul_(-0.5)
        return loss

    def generate_from_text(self, texts, dataset, noise=None):
        encoded = [dataset.train.encode_text(t) for t in texts]
        generated = self.generate_from_encoded_text(encoded, dataset, noise)
        return generated

    def generate_from_encoded_text(self, encoded, dataset, noise=None):
        with torch.no_grad():
            w_emb, s_emb = self.damsm.txt_enc(encoded)
            attn_mask = torch.tensor(encoded).to(self.device) == dataset.vocab[END_TOKEN]
            if noise is None:
                noise = torch.FloatTensor(len(encoded), D_Z, device=self.device)
                noise.data.normal_(0, 1)
            generated, att, mu, logvar = self.gen(noise, s_emb, w_emb, attn_mask)
        return generated

    def _save_generated(self, generated, epoch, dir):
        nb_samples = generated[0].size(0)
        save_dir = f'dir/epoch_{epoch}'
        os.makedirs(save_dir)
        for i in range(nb_samples):
            save_image(generated[0][i], f'{save_dir}/{i}_64')
            save_image(generated[1][i], f'{save_dir}/{i}_128')
            save_image(generated[2][i], f'{save_dir}/{i}_256')
