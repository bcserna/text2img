from src.config import DEVICE
from src.data import CUB
from src.damsm import DAMSM
from src.gan import AttnGAN
from src.generator import Generator
from src.discriminator import Discriminator, PatchDiscriminator


def train_gan(epochs, device=DEVICE):
    cub = CUB()
    damsm = DAMSM.load('l01992')
    generator = Generator(device)
    # discriminator = Discriminator(device)
    discriminator = PatchDiscriminator(device)
    gan = AttnGAN(damsm, generator, discriminator, device)
    metrics = gan.train(cub, epochs)
    return metrics
