import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import linalg
from scipy.stats import entropy
from itertools import cycle
from tqdm import tqdm

from src.config import GAN_BATCH, DEVICE, END_TOKEN, D_Z


class InceptionFrechetActivationHook:
    def __init__(self, inception_model):
        self.hook = inception_model.Mixed_7c.register_forward_hook(self.hook_fn)
        self.out = None

    def hook_fn(self, module, x, y):
        self.out = y

    def close(self):
        self.hook.remove()


def activation_statistics(inception_model, imgs, batch_size=32, device=DEVICE):
    with torch.no_grad():
        hook = InceptionFrechetActivationHook(inception_model)
        loader = DataLoader(imgs, batch_size=batch_size, shuffle=False, drop_last=False)
        activations = np.zeros((len(imgs), 2048), dtype=np.float32)
        for i, batch in enumerate(
                tqdm(loader, desc='Calculating inception activation statistics', dynamic_ncols=True, leave=False)):
            inception_model(torch.FloatTensor(batch).to(device))
            act = hook.out
            act = F.adaptive_avg_pool2d(act, (1, 1))
            act = torch.flatten(act, 1)
            act = act.cpu().numpy()
            activations[i * batch_size:i * batch_size + len(batch)] = act

        mu = np.mean(activations, axis=0)
        sig = np.cov(activations, rowvar=False)
        return mu, sig


def frechet_inception_distance(model, dataset, inception_model, batch_size=GAN_BATCH, device=DEVICE):
    nb_samples = len(dataset.train)
    real_imgs = [imgs[-1] for imgs, cap, label in
                 tqdm(dataset.train, desc='Loading training images', leave=False, dynamic_ncols=True)]
    fake_imgs = generate_test_samples(model, dataset, nb_samples, batch_size, device)
    mu_real, sig_real = activation_statistics(inception_model, real_imgs, batch_size, device)
    mu_fake, sig_fake = activation_statistics(inception_model, fake_imgs, batch_size, device)

    return frechet_dist(mu_real, sig_real, mu_fake, sig_fake)


def embed_captions(captions, encoder, dataset, device=DEVICE):
    encoder.eval()
    word_embs, sent_embs = encoder(captions)
    attn_mask = torch.tensor(captions).to(device) == dataset.vocab[END_TOKEN]
    return word_embs, sent_embs, attn_mask


def generate_test_samples(model, dataset, n_samples, batch_size=GAN_BATCH, device=DEVICE):
    with torch.no_grad():
        model.gen.eval()
        loader = cycle(DataLoader(dataset.test, batch_size=batch_size, shuffle=True, drop_last=False,
                                  collate_fn=dataset.collate_fn))
        generated_samples = np.zeros((n_samples, 3, 256, 256), dtype=np.float32)
        nb_generated = 0
        pbar = tqdm(loader, desc='Generating samples', dynamic_ncols=True, leave=False,
                    total=n_samples
                    # total=math.ceil(n_samples / batch_size)
                    )
        for batch in loader:
            word_embs, sent_embs, attn_mask = embed_captions(batch['caption'], model.damsm.txt_enc, dataset, device)
            l = sent_embs.size(0)

            # Generate images
            noise = torch.FloatTensor(l, D_Z).to(device)
            noise.data.normal_(0, 1)
            generated, att, mu, logvar = model.gen(noise, sent_embs, word_embs, attn_mask)
            generated = generated[-1].cpu().numpy()
            if nb_generated + l < n_samples:
                generated_samples[nb_generated:nb_generated + l] = generated
                nb_generated += l
                pbar.update(l)
            else:
                generated_samples[nb_generated:] = generated[:n_samples - nb_generated]
                break
        pbar.close()
    return generated_samples


def frechet_dist(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance
       The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
       and X_2 ~ N(mu_2, C_2) is
               d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
       Stable version by Dougal J. Sutherland.
       Params:
       -- mu1   : Numpy array containing the activations of a layer of the
                  inception net (like returned by the function 'get_predictions')
                  for generated samples.
       -- mu2   : The sample mean over activations, precalculated on an
                  representative data set.
       -- sigma1: The covariance matrix over activations for generated samples.
       -- sigma2: The covariance matrix over activations, precalculated on an
                  representative data set.
       Returns:
       --   : The Frechet Distance.
       """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def inception_score(gan, dataset, inception_model, batch_size=GAN_BATCH, samples=50000, splits=10, device=DEVICE):
    training = gan.gen.training
    with torch.no_grad():
        gan.gen.eval()
        generated = generate_test_samples(gan, dataset, samples, batch_size, device)
        loader = DataLoader(generated, batch_size=batch_size, drop_last=False, shuffle=False)
        inception_preds = np.zeros((samples, 1000), dtype=np.float32)
        preds = 0
        for x in tqdm(loader, desc='Computing inception score', leave=False, dynamic_ncols=True):
            x = F.interpolate(torch.FloatTensor(x).to(device), size=(299, 299), mode='bilinear',
                              align_corners=False)
            x = inception_model(x)
            x = F.softmax(x, dim=-1).data.cpu().numpy()

            l = len(x)
            inception_preds[preds:preds + l] = x
            preds += l

        scores = []
        split_size = samples // splits
        for s in range(splits):
            split_scores = []

            split = inception_preds[s * split_size: (s + 1) * split_size]
            p_y = np.mean(split, axis=0)
            for sample_pred in split:
                split_scores.append(entropy(sample_pred, p_y))

            scores.append(np.exp(np.mean(split_scores)))

        gan.gen.train(mode=training)
        return np.mean(scores), np.std(scores)
