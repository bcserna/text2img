import json
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
import time
import os
from tqdm import tqdm
from copy import deepcopy

from src.config import DEVICE, GAN_BATCH, GENERATOR_LR, DISCRIMINATOR_LR, D_Z, END_TOKEN, LAMBDA, GAN_MODEL_DIR, OUT_DIR
from src.util import rotate_tensor, init_weights, grad_norm, freeze_params_, pre_json_metrics
from src.generator import Generator
from src.discriminator import Discriminator


class AttnGAN:
    def __init__(self, damsm, device=DEVICE):
        self.gen = Generator(device)
        self.disc = Discriminator(device)
        self.damsm = damsm.to(device)
        self.damsm.txt_enc.eval(), self.damsm.img_enc.eval()
        freeze_params_(self.damsm.txt_enc), freeze_params_(self.damsm.img_enc)

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

    def train(self, dataset, epoch, batch_size=GAN_BATCH, test_sample_every=5, hist_avg=True, evaluator=None):
        start_time = time.strftime("%Y-%m-%d-%H-%M", time.gmtime())
        os.makedirs(f'{OUT_DIR}/{start_time}')

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
            'IS': [],
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

        if evaluator is not None:
            evaluator = evaluator(dataset, self.damsm.img_enc.inception_model, batch_size, self.device)

        noise = torch.FloatTensor(batch_size, D_Z).to(self.device)
        gen_updates = 0
        for e in tqdm(range(epoch), desc='Epochs', dynamic_ncols=True):
            self.gen.train(), self.disc.train()
            g_loss = 0
            w_loss = 0
            s_loss = 0
            kl_loss = 0
            g_stage_loss = np.zeros(3, dtype=float)
            d_loss = np.zeros(3, dtype=float)
            real_acc = np.zeros(3, dtype=float)
            fake_acc = np.zeros(3, dtype=float)
            mismatched_acc = np.zeros(3, dtype=float)
            uncond_real_acc = np.zeros(3, dtype=float)
            uncond_fake_acc = np.zeros(3, dtype=float)
            disc_skips = np.zeros(3, dtype=int)

            train_pbar = tqdm(train_loader, desc='Training', leave=False, dynamic_ncols=True)
            for batch in train_pbar:
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

                d_grad_norm = [grad_norm(d) for d in self.discriminators]

                d_loss += batch_d_loss
                real_acc += batch_real_acc
                fake_acc += batch_fake_acc
                mismatched_acc += batch_mismatched_acc
                uncond_real_acc += batch_uncond_real_acc
                uncond_fake_acc += batch_uncond_fake_acc
                disc_skips += batch_disc_skips

                # Generator loss
                batch_g_losses = self.generator_step(generated, word_embs, sent_embs, mu, logvar, batch['label'])
                g_total, batch_g_stage_loss, batch_w_loss, batch_s_loss, batch_kl_loss = batch_g_losses
                g_stage_loss += batch_g_stage_loss
                w_loss += batch_w_loss
                s_loss += batch_s_loss
                kl_loss += batch_kl_loss
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

                # train_pbar.set_description(f'Training (G: {avg_g_loss:05.4f}  '
                #                            f'D64: {batch_d_loss[0]:05.4f}  '
                #                            f'D128: {batch_d_loss[1]:05.4f}  '
                #                            f'D256: {batch_d_loss[2]:05.4f})')
                train_pbar.set_description(f'Training (G: {grad_norm(self.gen):.2f}  '
                                           f'D64: {d_grad_norm[0]:.2f}  '
                                           f'D128: {d_grad_norm[1]:.2f}  '
                                           f'D256: {d_grad_norm[2]:.2f})')

            batches = len(train_loader)

            g_loss /= batches
            g_stage_loss /= batches
            w_loss /= batches
            s_loss /= batches
            kl_loss /= batches
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
                generated_samples = [resolution.unsqueeze(0) for resolution in self.sample_test_set(dataset)]
                self._save_generated(generated_samples, e, f'{OUT_DIR}/{start_time}')

                if evaluator is not None:
                    scores = evaluator.evaluate(self)
                    for k, v in scores.items():
                        metrics[k].append(v)
                        tqdm.write(f'{k}: {v:.2f}')

            tqdm.write(f'Generator avg loss: total({g_loss:.3f})  '
                       f'stage0({g_stage_loss[0]:.3f})  stage1({g_stage_loss[1]:.3f})  stage2({g_stage_loss[2]:.3f})  '
                       f'w({w_loss:.3f})  s({s_loss:.3f})  kl({kl_loss:.3f})')

            for i, _ in enumerate(self.discriminators):
                tqdm.write(f'Discriminator{i} avg: '
                           f'loss({d_loss[i]:.3f})  '
                           f'r-acc({real_acc[i]:.3f})  '
                           f'f-acc({fake_acc[i]:.3f})  '
                           f'm-acc({mismatched_acc[i]:.3f})  '
                           f'ur-acc({uncond_real_acc[i]:.3f})  '
                           f'uf-acc({uncond_fake_acc[i]:.3f})  '
                           f'skips({disc_skips[i]})')

        return metrics

    def sample_test_set(self, dataset, nb_samples=8, nb_captions=2, noise_variations=2):
        subset = dataset.test
        sample_indices = np.random.choice(len(subset), nb_samples, replace=False)
        cap_indices = np.random.choice(10, nb_captions, replace=False)
        texts = [subset.data[f'caption_{cap_idx}'].iloc[sample_idx]
                 for sample_idx in sample_indices
                 for cap_idx in cap_indices]

        generated_samples = [self.generate_from_text(texts, dataset) for _ in range(noise_variations)]
        combined_img64 = torch.FloatTensor()
        combined_img128 = torch.FloatTensor()
        combined_img256 = torch.FloatTensor()

        for noise_variant in generated_samples:
            noise_var_img64 = torch.FloatTensor()
            noise_var_img128 = torch.FloatTensor()
            noise_var_img256 = torch.FloatTensor()
            for i in range(nb_samples):
                # rows: samples, columns: captions * noise variants
                row64 = torch.cat([noise_variant[0][i * nb_captions + j] for j in range(nb_captions)], dim=-1).cpu()
                row128 = torch.cat([noise_variant[1][i * nb_captions + j] for j in range(nb_captions)], dim=-1).cpu()
                row256 = torch.cat([noise_variant[2][i * nb_captions + j] for j in range(nb_captions)], dim=-1).cpu()
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
        self.gen.zero_grad()
        avg_stage_g_loss = [0, 0, 0]

        local_features, global_features = self.damsm.img_enc(generated_imgs[-1])
        batch_size = sent_embs.size(0)
        match_labels = torch.LongTensor(range(batch_size)).to(self.device)

        w1_loss, w2_loss, _ = self.damsm.words_loss(local_features, word_embs, class_labels, match_labels)
        w_loss = (w1_loss + w2_loss) * LAMBDA

        s1_loss, s2_loss = self.damsm.sentence_loss(global_features, sent_embs, class_labels, match_labels)
        s_loss = (s1_loss + s2_loss) * LAMBDA

        kl_loss = self.KL_loss(mu, logvar)

        g_total = w_loss + s_loss + kl_loss

        for i, d in enumerate(self.discriminators):
            features = d(generated_imgs[i])
            fake_logits = d.logit(features, sent_embs)

            real_labels = torch.ones_like(fake_logits).to(self.device)

            disc_error = F.binary_cross_entropy_with_logits(fake_logits, real_labels)

            uncond_fake_logits = d.logit(features)
            uncond_disc_error = F.binary_cross_entropy_with_logits(uncond_fake_logits, real_labels)

            stage_loss = disc_error + uncond_disc_error
            avg_stage_g_loss[i] = stage_loss.item() / batch_size
            g_total += stage_loss

        g_total.backward()
        self.gen_optimizer.step()

        return g_total, avg_stage_g_loss, w_loss.item() / batch_size, s_loss.item() / batch_size, kl_loss.item()

    def discriminator_step(self, real_imgs, generated_imgs, sent_embs, label_smoothing, skip_acc_threshold=0.9,
                           p_flip=0.05, halting=False):
        self.disc.zero_grad()
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

            # real_labels = torch.ones_like(real_logits, dtype=torch.float).to(self.device)
            real_labels = torch.full_like(real_logits, 1 - label_smoothing).to(self.device)
            fake_labels = torch.zeros_like(real_logits, dtype=torch.float).to(self.device)

            # flip_mask = torch.Tensor(real_labels.size()).bernoulli_(p_flip).type(torch.bool)
            # real_labels[flip_mask], fake_labels[flip_mask] = fake_labels[flip_mask], real_labels[flip_mask]

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

            error = (real_error + uncond_real_error) / 2 + (fake_error + uncond_fake_error + mismatched_error) / 3

            if not halting or fake_accuracy[i] + real_accuracy[i] < skip_acc_threshold * 2:
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

    def _save_generated(self, generated, epoch, out_dir=OUT_DIR):
        nb_samples = generated[0].size(0)
        save_dir = f'{out_dir}/epoch_{epoch:03}'
        os.makedirs(save_dir)

        for i in range(nb_samples):
            save_image(generated[0][i], f'{save_dir}/{i}_64.jpg', normalize=True, range=(-1, 1))
            save_image(generated[1][i], f'{save_dir}/{i}_128.jpg', normalize=True, range=(-1, 1))
            save_image(generated[2][i], f'{save_dir}/{i}_256.jpg', normalize=True, range=(-1, 1))

    def save(self, name, save_dir=GAN_MODEL_DIR, metrics=None):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.gen.state_dict(), f'{save_dir}/{name}_generator.pt')
        torch.save(self.disc.state_dict(), f'{save_dir}/{name}_discriminator.pt')
        if metrics is not None:
            with open(f'{save_dir}/{name}_metrics.json', 'w') as f:
                metrics = pre_json_metrics(metrics)
                json.dump(metrics, f)

    def load_(self, name, load_dir=GAN_MODEL_DIR):
        self.gen.load_state_dict(torch.load(f'{load_dir}/{name}_generator.pt'))
        self.disc.load_state_dict(torch.load(f'{load_dir}/{name}_discriminator.pt'))
        self.gen.eval(), self.disc.eval()

    @staticmethod
    def load(name, damsm, load_dir=GAN_MODEL_DIR, device=DEVICE):
        attngan = AttnGAN(damsm, device=device)
        attngan.load_(name, load_dir)
        return attngan

    def validate_test_set(self, dataset, batch_size=GAN_BATCH, save_dir=f'{OUT_DIR}/test_samples'):
        os.makedirs(save_dir, exist_ok=True)

        loader = DataLoader(dataset.test, batch_size=batch_size, shuffle=True, drop_last=False,
                            collate_fn=dataset.collate_fn)
        loader = tqdm(loader, dynamic_ncols=True, leave=True, desc='Generating samples for test set')

        self.gen.eval()
        with torch.no_grad():
            i = 0
            for batch in loader:
                word_embs, sent_embs = self.damsm.txt_enc(batch['caption'])
                attn_mask = torch.tensor(batch['caption']).to(self.device) == dataset.vocab[END_TOKEN]
                noise = torch.FloatTensor(len(batch['caption']), D_Z).to(self.device)
                noise.data.normal_(0, 1)
                generated, att, mu, logvar = self.gen(noise, sent_embs, word_embs, attn_mask)

                for img in generated[-1]:
                    save_image(img, f'{save_dir}/{i}.jpg', normalize=True, range=(-1, 1))
                    i += 1
