import json

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
import time
import os
from tqdm import tqdm
from copy import deepcopy

from src.config import DEVICE, GAN_BATCH, GENERATOR_LR, DISCRIMINATOR_LR, D_Z, END_TOKEN, LAMBDA, MODEL_DIR
from src.util import rotate_tensor, init_weights
from src.evaluation import inception_score, frechet_inception_distance


class AttnGAN:
    def __init__(self, damsm, generator, discriminator, device=DEVICE):
        self.gen = generator.to(device)
        self.disc = discriminator.to(device)
        self.damsm = damsm.to(device)
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

    def train(self, dataset, epoch, batch_size=GAN_BATCH, test_sample_every=3, hist_avg=True, fid_evaluator=None):
        start_time = time.strftime("%Y-%m-%d-%H-%M", time.gmtime())
        os.makedirs(start_time)

        if hist_avg:
            avg_g_params = deepcopy(list(p.data for p in self.gen.parameters()))

        loader_config = {
            'batch_size': batch_size,
            'shuffle': True,
            'drop_last': True,
            'collate_fn': dataset.collate_fn
        }
        train_loader = DataLoader(dataset.train, **loader_config)

        metrics = {
            'FID': [],
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

        if self.fid_evaulator is not None:
            fid = fid_evaluator(dataset, self.damsm.img_enc.inception_model, batch_size, self.device)

        noise = torch.FloatTensor(batch_size, D_Z).to(self.device)
        gen_updates = 0
        for e in tqdm(range(epoch), desc='Epochs', dynamic_ncols=True):
            self.gen.train(), self.disc.train()
            g_loss = 0
            d_loss = np.zeros(3, dtype=float)
            real_acc = np.zeros(3, dtype=float)
            fake_acc = np.zeros(3, dtype=float)
            mismatched_acc = np.zeros(3, dtype=float)
            uncond_real_acc = np.zeros(3, dtype=float)
            uncond_fake_acc = np.zeros(3, dtype=float)
            disc_skips = np.zeros(3, dtype=int)

            train_pbar = tqdm(train_loader, desc='Training', leave=False, dynamic_ncols=True)
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
                    real_imgs, generated, sent_embs, 0.1)

                d_loss += batch_d_loss
                real_acc += batch_real_acc
                fake_acc += batch_fake_acc
                mismatched_acc += batch_mismatched_acc
                uncond_real_acc += batch_uncond_real_acc
                uncond_fake_acc += batch_uncond_fake_acc
                disc_skips += batch_disc_skips

                # Generator loss
                g_total = self.generator_step(generated, word_embs, sent_embs, mu, logvar, batch['label'])
                gen_updates += 1

                avg_g_loss = g_total.item() / batch_size
                g_loss += avg_g_loss

                if hist_avg:
                    for p, avg_p in zip(self.gen.parameters(), avg_g_params):
                        avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_updates % 1000 == 0:
                    tqdm.write('Replacing generator weights with their moving average')
                    for p, avg_p in zip(self.gen.parameters(), avg_g_params):
                        p.data.copy_(avg_p)

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

            sep = '_' * 10
            tqdm.write(f'{sep}Epoch {e}{sep}')

            if e % test_sample_every == 0:
                self.gen.eval()
                texts = [dataset.test.data['caption_0'].iloc[sample_idx] for sample_idx in range(2)]
                # generated_samples = self.generate_from_text(texts, dataset)
                generated_samples = [resolution.unsqueeze(0) for resolution in self.sample_test_set(dataset)]
                self._save_generated(generated_samples, e, start_time)

                # inc_score = inception_score(self, dataset, self.damsm.img_enc.inception_model, batch_size,
                #                             device=self.device)
                # metrics['inception'].append(inc_score)
                # tqdm.write(f'Inception score: {inc_score[0]:02.2f} +- {inc_score[1]:02.2f}')

                if fid_evaluator is not None:
                    fid_score = fid.evaluate(self)
                    metrics['FID'].append(fid_score)
                    tqdm.write(f'FID: {fid_score:04.2f}')

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

    def sample_test_set(self, dataset, nb_samples=4, nb_captions=2, noise_variations=2):
        texts = [dataset.test.data[f'caption_{cap_idx}'].iloc[sample_idx]
                 for sample_idx in range(nb_samples)
                 for cap_idx in range(nb_captions)]

        generated_samples = [self.generate_from_text(texts, dataset) for _ in range(noise_variations)]

        # combined_img64 = torch.FloatTensor(3, nb_samples * 64, nb_captions * noise_variations * 64)
        combined_img64 = torch.FloatTensor()
        combined_img128 = torch.FloatTensor()
        combined_img256 = torch.FloatTensor()

        for noise_variant in generated_samples:
            noise_var_img64 = torch.FloatTensor()
            noise_var_img128 = torch.FloatTensor()
            noise_var_img256 = torch.FloatTensor()
            for i in range(nb_samples):
                # rows: samples, columns: captions * noise variants
                row64 = torch.cat([noise_variant[0][i * nb_captions + j] for j in range(nb_captions)], dim=-1)
                row128 = torch.cat([noise_variant[1][i * nb_captions + j] for j in range(nb_captions)], dim=-1)
                row256 = torch.cat([noise_variant[2][i * nb_captions + j] for j in range(nb_captions)], dim=-1)
                noise_var_img64 = torch.cat([noise_var_img64, row64], dim=-2)
                noise_var_img128 = torch.cat([noise_var_img128, row128], dim=-2)
                noise_var_img256 = torch.cat([noise_var_img256, row256], dim=-2)
            combined_img64 = torch.cat([combined_img64, noise_var_img64], dim=-1)
            combined_img128 = torch.cat([combined_img128, noise_var_img128], dim=-1)
            combined_img256 = torch.cat([combined_img256, noise_var_img256], dim=-1)

        return combined_img64, combined_img128, combined_img256

    @staticmethod
    def KL_loss(mu, logvar):
        loss = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        loss = torch.mean(loss).mul_(-0.5)
        return loss

    def generator_step(self, generated_imgs, word_embs, sent_embs, mu, logvar, class_labels):
        local_features, global_features = self.damsm.img_enc(generated_imgs[-1])
        batch_size = sent_embs.size(0)
        match_labels = torch.LongTensor(range(batch_size)).requires_grad_(False).to(self.device)

        w1_loss, w2_loss, _ = self.damsm.words_loss(local_features, word_embs, class_labels, match_labels)
        w_loss = (w1_loss + w2_loss) * LAMBDA

        s1_loss, s2_loss = self.damsm.sentence_loss(global_features, sent_embs, class_labels, match_labels)
        s_loss = (s1_loss + s2_loss) * LAMBDA

        kl_loss = self.KL_loss(mu, logvar)

        g_loss = w_loss + s_loss + kl_loss

        for i, d in enumerate(self.discriminators):
            features = d(generated_imgs[i])
            fake_logits = d.logit(features, sent_embs)

            real_labels = torch.Tensor(fake_logits.size()).fill_(1).requires_grad_(False).to(self.device)

            disc_error = F.binary_cross_entropy_with_logits(fake_logits, real_labels)

            uncond_fake_logits = d.logit(features)
            uncond_disc_error = F.binary_cross_entropy_with_logits(uncond_fake_logits, real_labels)

            g_loss += disc_error + uncond_disc_error

        g_loss.backward()
        self.gen_optimizer.step()

        return g_loss

    def discriminator_step(self, real_imgs, generated_imgs, sent_embs, label_smoothing, skip_acc_threshold=0.9,
                           p_flip=0.05):
        batch_size = sent_embs.size(0)

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

            real_labels = torch.Tensor(real_logits.size()).fill_(1).requires_grad_(False).to(self.device)
            fake_labels = torch.Tensor(real_logits.size()).fill_(0).requires_grad_(False).to(self.device)

            real_labels = real_labels - label_smoothing
            fake_labels = fake_labels + label_smoothing

            flip_mask = torch.Tensor(real_labels.size()).bernoulli_(p_flip).type(torch.bool)
            real_labels[flip_mask], fake_labels[flip_mask] = fake_labels[flip_mask], real_labels[flip_mask]

            real_error = F.binary_cross_entropy_with_logits(real_logits, real_labels)
            # Real images should be classified as real
            real_accuracy[i] = (real_logits >= 0).sum().item() / real_logits.numel()

            fake_logits = d.logit(fake_features, sent_embs)
            fake_error = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
            # Generated images should be classified as fake
            fake_accuracy[i] = (fake_logits < 0).sum().item() / fake_logits.numel()

            mismatched_logits = d.logit(real_features, rotate_tensor(sent_embs, 1))
            mismatched_error = F.binary_cross_entropy_with_logits(mismatched_logits, fake_labels)
            # Images with mismatched descriptions should be classified as fake
            mismatched_accuracy[i] = (mismatched_logits < 0).sum().item() / mismatched_logits.numel()

            uncond_real_logits = d.logit(real_features)
            uncond_real_error = F.binary_cross_entropy_with_logits(uncond_real_logits, real_labels)
            uncond_real_accuracy[i] = (uncond_real_logits >= 0).sum().item() / uncond_real_logits.numel()

            uncond_fake_logits = d.logit(fake_features)
            uncond_fake_error = F.binary_cross_entropy_with_logits(uncond_fake_logits, fake_labels)
            uncond_fake_accuracy[i] = (uncond_fake_logits < 0).sum().item() / uncond_fake_logits.numel()

            error = ((real_error + uncond_real_error) / 2 + fake_error + uncond_fake_error + mismatched_error) / 3

            if fake_accuracy[i] + real_accuracy[i] < skip_acc_threshold * 2:
                error.backward()
                self.disc_optimizers[i].step()
            else:
                skipped[i] = 1
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

    def save(self, name, save_dir=MODEL_DIR, metrics=None):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.gen.state_dict(), f'{save_dir}/{name}_generator.pt')
        torch.save(self.disc.state_dict(), f'{save_dir}/{name}_discriminator.pt')
        if metrics is not None:
            with open(f'{save_dir}/{name}_metrics.json', 'w') as f:
                json.dump(metrics, f)

    def load_(self, name, load_dir=MODEL_DIR):
        self.gen.load_state_dict(torch.load(f'{load_dir}/{name}_generator.pt'))
        self.disc.load_state_dict(torch.load(f'{load_dir}/{name}_discriminator.pt'))
        self.gen.eval(), self.disc.eval()

    @staticmethod
    def load(name, damsm, generator_type, discriminator_type, load_dir=MODEL_DIR):
        attngan = AttnGAN(damsm, generator_type(), discriminator_type())
        attngan.load_(name, load_dir)
        return attngan
