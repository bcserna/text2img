import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
import time
import os
from tqdm import tqdm

from src.config import DEVICE, GAN_BATCH, GENERATOR_LR, DISCRIMINATOR_LR, D_Z, END_TOKEN, LAMBDA, MODEL_DIR
from src.discriminator import Discriminator
from src.generator import Generator
from src.util import rotate_tensor, init_weights, inception_score


class AttnGAN:
    def __init__(self, damsm, device=DEVICE):
        self.gen = Generator().to(device)
        self.disc = Discriminator().to(device)
        self.damsm = damsm
        self.damsm.txt_enc.eval(), self.damsm.img_enc.eval()
        self.device = device
        self.gen.apply(init_weights), self.disc.apply(init_weights)

        self.gen_optimizer = torch.optim.Adam(self.gen.parameters(),
                                              lr=GENERATOR_LR,
                                              betas=(0.5, 0.999))

        self.discriminators = [self.disc.d64, self.disc.d128, self.disc.d256]
        self.disc_optimizers = [torch.optim.Adam(d.parameters(),
                                                 lr=DISCRIMINATOR_LR,
                                                 betas=(0.5, 0.999))
                                for d in self.discriminators]

    def train(self, dataset, epoch, batch_size=GAN_BATCH, test_sample_every=3, nb_test_samples=2):
        start_time = time.strftime("%Y-%m-%d-%H-%M", time.gmtime())
        os.makedirs(start_time)

        loader_config = {
            'batch_size': batch_size,
            'shuffle': True,
            'drop_last': True,
            'collate_fn': dataset.collate_fn
        }
        train_loader = DataLoader(dataset.train, **loader_config)

        metrics = {
            'inception': [],
            'loss': {
                'g': [],
                'd': []
            },
            'accuracy': {
                'real': [],
                'fake': [],
                'mismatched': [],
                'unconditional_real': [],
                'unconditional_fake': []
            }
        }

        real_labels = nn.Parameter(torch.FloatTensor(batch_size).fill_(0), requires_grad=False).to(self.device)
        fake_labels = nn.Parameter(torch.FloatTensor(batch_size).fill_(1), requires_grad=False).to(self.device)
        match_labels = nn.Parameter(torch.LongTensor(range(batch_size)), requires_grad=False).to(self.device)

        noise = torch.FloatTensor(batch_size, D_Z).to(self.device)
        for e in tqdm(range(epoch), desc='Epochs'):
            self.gen.train(), self.disc.train()
            g_loss = 0
            d_loss = np.zeros(3, dtype=float)
            real_acc = np.zeros(3, dtype=float)
            fake_acc = np.zeros(3, dtype=float)
            mismatched_acc = np.zeros(3, dtype=float)
            uncond_real_acc = np.zeros(3, dtype=float)
            uncond_fake_acc = np.zeros(3, dtype=float)
            disc_skips = np.zeros(3, dtype=int)

            train_pbar = tqdm(train_loader, desc='Training', leave=False)
            for batch in train_pbar:
                self.gen.zero_grad(), self.disc.zero_grad()
                real_imgs = [batch['img64'], batch['img128'], batch['img256']]

                with torch.no_grad():
                    word_embs, sent_embs = self.damsm.txt_enc(batch['caption'])
                attn_mask = torch.tensor(batch['caption']).to(self.device) == dataset.vocab[END_TOKEN]

                # Generate images
                noise.data.normal_(0, 1)
                generated, att, mu, logvar = self.gen(noise, sent_embs, word_embs, attn_mask)

                # Discriminator loss (with label smoothing)
                batch_d_loss, batch_real_acc, batch_fake_acc, batch_mismatched_acc, batch_uncond_real_acc, batch_uncond_fake_acc, batch_disc_skips = self.discriminator_step(
                    real_imgs, generated, sent_embs, real_labels, fake_labels, 0.1, 0)

                d_loss += batch_d_loss
                real_acc += batch_real_acc
                fake_acc += batch_fake_acc
                mismatched_acc += batch_mismatched_acc
                uncond_real_acc += batch_uncond_real_acc
                uncond_fake_acc += batch_uncond_fake_acc
                disc_skips += batch_disc_skips

                # Generator loss
                g_total = self.generator_step(generated, word_embs, sent_embs, mu, logvar, real_labels, batch['label'],
                                              match_labels)
                avg_g_loss = g_total.item() / batch_size
                g_loss += avg_g_loss

                train_pbar.set_description(f'Training (G: {avg_g_loss:05.4f}  '
                                           f'D64: {batch_d_loss[0]:05.4f}  '
                                           f'D128: {batch_d_loss[1]:05.4f}  '
                                           f'D256: {batch_d_loss[2]:05.4f})')

            batches = len(train_loader)

            g_loss /= batches

            d_loss /= batches
            real_acc /= batches
            fake_acc /= batches
            mismatched_acc /= batches
            uncond_real_acc /= batches
            uncond_fake_acc /= batches

            metrics['loss']['g'].append(g_loss)
            metrics['loss']['d'].append(d_loss)
            metrics['accuracy']['real'].append(real_acc)
            metrics['accuracy']['fake'].append(fake_acc)
            metrics['accuracy']['mismatched'].append(mismatched_acc)
            metrics['accuracy']['unconditional_real'].append(uncond_real_acc)
            metrics['accuracy']['unconditional_fake'].append(uncond_fake_acc)

            if e % test_sample_every == 0:
                texts = [dataset.test.data['caption_0'].iloc[i] for i in range(nb_test_samples)]
                generated_samples = self.generate_from_text(texts, dataset)
                self._save_generated(generated_samples, e, start_time)
                inc_score = inception_score(self, dataset, self.damsm.img_enc.inception_model, batch_size,
                                            device=self.device)
                metrics['inception'].append(inc_score)
            else:
                inc_score = None

            sep = '_' * 10
            tqdm.write(f'{sep}Epoch {e}{sep}')
            if inc_score is not None:
                tqdm.write(f'Inception score: {inc_score[0]:02.2f} +- {inc_score[1]:02.2f}')
            tqdm.write(f'Generator avg loss: {g_loss:05.4f}')

            for i, _ in enumerate(self.discriminators):
                tqdm.write(f'Discriminator{i} avg: '
                           f'loss({d_loss[i]:05.4f})  '
                           f'r-acc({real_acc[i]:04.3f})  '
                           f'f-acc({fake_acc[i]:04.3f})  '
                           f'm-acc({mismatched_acc[i]:04.3f})  '
                           f'ur-acc({uncond_real_acc[i]:04.3f})  '
                           f'uf-acc({uncond_fake_acc[i]:04.3f})  '
                           f'skips({disc_skips[i]})')

        return metrics

    @staticmethod
    def KL_loss(mu, logvar):
        loss = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        loss = torch.mean(loss).mul_(-0.5)
        return loss

    def generator_step(self, generated_imgs, word_embs, sent_embs, mu, logvar, real_labels, class_labels, match_labels):
        local_features, global_features = self.damsm.img_enc(generated_imgs[-1])

        w1_loss, w2_loss, _ = self.damsm.words_loss(local_features, word_embs, class_labels, match_labels)
        w_loss = (w1_loss + w2_loss) * LAMBDA

        s1_loss, s2_loss = self.damsm.sentence_loss(global_features, sent_embs, class_labels, match_labels)
        s_loss = (s1_loss + s2_loss) * LAMBDA

        kl_loss = self.KL_loss(mu, logvar)

        g_loss = w_loss + s_loss + kl_loss

        for i, d in enumerate(self.discriminators):
            features = d(generated_imgs[i])
            fake_logits = d.logit(features, sent_embs)
            disc_error = nn.functional.binary_cross_entropy_with_logits(fake_logits, real_labels)

            uncond_fake_logits = d.logit(features)
            uncond_disc_error = nn.functional.binary_cross_entropy_with_logits(uncond_fake_logits, real_labels)

            g_loss += disc_error + uncond_disc_error

        g_loss.backward()
        self.gen_optimizer.step()

        return g_loss

    def discriminator_step(self, real_imgs, generated_imgs, sent_embs, real_labels, fake_labels,
                           real_smoothing, fake_smoothing, skip_acc_threshold=0.9):
        batch_size = real_labels.size(0)

        smooth_real_labels = real_labels + real_smoothing
        smooth_fake_labels = fake_labels + fake_smoothing
        # Add label noise
        p_flip = 0.05
        flip_mask = torch.zeros(batch_size).bernoulli_(p_flip).type(torch.bool)
        smooth_real_labels[flip_mask], smooth_fake_labels[flip_mask] = smooth_fake_labels[flip_mask], \
                                                                       smooth_real_labels[flip_mask]

        avg_d_loss = [0, 0, 0]
        real_accuracy = [0, 0, 0]
        fake_accuracy = [0, 0, 0]
        mismatched_accuracy = [0, 0, 0]
        uncond_real_accuracy = [0, 0, 0]
        uncond_fake_accuracy = [0, 0, 0]
        skipped = [0, 0, 0]

        for i, d in enumerate(self.discriminators):
            real_features = d(real_imgs[i].to(self.device))
            fake_features = d(generated_imgs[i].detach())

            real_logits = d.logit(real_features, sent_embs)
            # real_error = nn.functional.binary_cross_entropy(real_logits, real_labels)
            real_error = nn.functional.binary_cross_entropy_with_logits(real_logits, smooth_real_labels)
            # Real images should be classified as real
            real_accuracy[i] = (real_logits < 0).sum().item() / batch_size

            fake_logits = d.logit(fake_features, sent_embs)
            # fake_error = nn.functional.binary_cross_entropy(fake_logits, fake_labels)
            fake_error = nn.functional.binary_cross_entropy_with_logits(fake_logits, smooth_fake_labels)
            # Generated images should be classified as fake
            fake_accuracy[i] = (fake_logits >= 0).sum().item() / batch_size

            mismatched_logits = d.logit(real_features, rotate_tensor(sent_embs, 1))
            # mismatched_error = nn.functional.binary_cross_entropy(mismatched_logits, fake_labels)
            mismatched_error = nn.functional.binary_cross_entropy_with_logits(mismatched_logits, smooth_fake_labels)
            # Images with mismatched descriptions should be classified as fake
            mismatched_accuracy[i] = (mismatched_logits >= 0).sum().item() / batch_size

            uncond_real_logits = d.logit(real_features)
            # uncond_real_error = nn.functional.binary_cross_entropy(uncond_real_logits, real_labels)
            uncond_real_error = nn.functional.binary_cross_entropy_with_logits(uncond_real_logits, smooth_real_labels)
            uncond_real_accuracy[i] = (uncond_real_logits < 0).sum().item() / batch_size

            uncond_fake_logits = d.logit(fake_features)
            # uncond_fake_error = nn.functional.binary_cross_entropy(uncond_fake_logits, fake_labels)
            uncond_fake_error = nn.functional.binary_cross_entropy_with_logits(uncond_fake_logits, smooth_fake_labels)
            uncond_fake_accuracy[i] = (uncond_fake_logits >= 0).sum().item() / batch_size

            error = ((real_error + uncond_real_error) / 2 + fake_error + uncond_fake_error + mismatched_error) / 3
            # if fake_accuracy[i] < skip_acc_threshold or real_accuracy[i] < 1 - skip_acc_threshold:
            error.backward()
            self.disc_optimizers[i].step()
            # else:
            #     skipped[i] = 1
            avg_d_loss[i] = error.item() / batch_size

        return avg_d_loss, real_accuracy, fake_accuracy, mismatched_accuracy, uncond_real_accuracy, uncond_fake_accuracy, skipped

    def generate_from_text(self, texts, dataset, noise=None):
        encoded = [dataset.train.encode_text(t) for t in texts]
        generated = self.generate_from_encoded_text(encoded, dataset, noise)
        return generated

    def generate_from_encoded_text(self, encoded, dataset, noise=None):
        with torch.no_grad():
            w_emb, s_emb = self.damsm.txt_enc(encoded)
            attn_mask = torch.tensor(encoded).to(self.device) == dataset.vocab[END_TOKEN]
            if noise is None:
                noise = torch.FloatTensor(len(encoded), D_Z).to(self.device)
                noise.data.normal_(0, 1)
            generated, att, mu, logvar = self.gen(noise, s_emb, w_emb, attn_mask)
        return generated

    def _save_generated(self, generated, epoch, dir):
        nb_samples = generated[0].size(0)
        save_dir = f'{dir}/epoch_{epoch:03}'
        os.makedirs(save_dir)

        for i in range(nb_samples):
            save_image(generated[0][i], f'{save_dir}/{i}_64.jpg')
            save_image(generated[1][i], f'{save_dir}/{i}_128.jpg')
            save_image(generated[2][i], f'{save_dir}/{i}_256.jpg')

    def save(self, name, save_dir=MODEL_DIR):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.gen.state_dict(), f'{save_dir}/{name}_generator.pt')
        torch.save(self.disc.state_dict(), f'{save_dir}/{name}_discriminator.pt')

    def load_(self, name, load_dir=MODEL_DIR):
        self.gen.load_state_dict(torch.load(f'{load_dir}/{name}_generator.pt'))
        self.disc.load_state_dict(torch.load(f'{load_dir}/{name}_discriminator.pt'))
        self.gen.eval(), self.disc.eval()

    @staticmethod
    def load(name, damsm, load_dir=MODEL_DIR):
        attngan = AttnGAN(damsm)
        attngan.load_(name, load_dir)
        return attngan
