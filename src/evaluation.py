import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import linalg
from scipy.stats import entropy
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


def activation_statistics(inception_model, imgs, batch_size=32):
    with torch.no_grad():
        hook = InceptionFrechetActivationHook(inception_model)
        loader = DataLoader(imgs, batch_size=batch_size, shuffle=False, drop_last=False)
        activations = np.zeros((len(imgs), 2048), dtype=np.float32)
        for i, batch in enumerate(tqdm(loader, desc='Calculating inception activation statistics', dynamic_ncols=True)):
            inception_model(batch)
            act = hook.out
            act = F.adaptive_avg_pool2d(act, (1, 1))
            act = torch.flatten(act, 1)
            act = act.cpu().numpy()
            activations[i * batch_size:i * batch_size + len(batch)] = act

        mu = np.mean(activations, axis=0)
        sig = np.cov(activations, rowvar=False)
        return mu, sig


def frechet_inception_distance(inception_model, real_imgs, fake_imgs):
    mu_real, sig_real = activation_statistics(inception_model, real_imgs)
    mu_fake, sig_fake = activation_statistics(inception_model, fake_imgs)

    return frechet_dist(mu_real, sig_real, mu_fake, sig_fake)


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
        inception_preds = np.zeros((samples, 1000))

        loader = DataLoader(dataset.test, batch_size=batch_size, shuffle=True, drop_last=True,
                            collate_fn=dataset.collate_fn)

        epochs = math.ceil(samples / (len(loader) * batch_size))
        nb_generated = 0
        for _ in tqdm(range(epochs), desc='Generating samples for inception score', dynamic_ncols=True):
            for batch in loader:
                word_embs, sent_embs = gan.damsm.txt_enc(batch['caption'])
                attn_mask = torch.tensor(batch['caption']).to(device) == dataset.vocab[END_TOKEN]

                # Generate images
                noise = torch.FloatTensor(batch_size, D_Z).to(device)
                noise.data.normal_(0, 1)
                generated, att, mu, logvar = gan.gen(noise, sent_embs, word_embs, attn_mask)
                x = generated[-1]
                x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
                x = inception_model(x)
                x = F.softmax(x, dim=-1).data.cpu().numpy()

                samples_left = samples - nb_generated
                if samples_left < batch_size:
                    inception_preds[nb_generated:] = x[:samples_left]
                    break
                else:
                    inception_preds[nb_generated:nb_generated + batch_size] = x
                    nb_generated += batch_size

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
