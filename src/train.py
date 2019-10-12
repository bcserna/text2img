from src.config import DEVICE
from src.data import CUB
from src.damsm import DAMSM
from src.evaluation import FIDEvaluator
from src.gan import AttnGAN
from src.generator import Generator
from src.discriminator import Discriminator, PatchDiscriminator


def train_gan(epochs, name, gan=None, device=DEVICE):
    cub = CUB()
    damsm = DAMSM.load('l01992')
    if gan is None:
        generator = Generator(device)
        discriminator = Discriminator(device)
        # discriminator = PatchDiscriminator(device)
        gan = AttnGAN(damsm, generator, discriminator, device)
    metrics = gan.train(cub, epochs, fid_evaluator=FIDEvaluator)
    gan.save(name, metrics=metrics)
    return gan, metrics


if __name__ == '__main__':
    import plac
    plac.call(train_gan)
