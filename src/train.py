import click

from src.config import DEVICE
from src.data import CUB
from src.damsm import DAMSM
from src.evaluation import FIDEvaluator
from src.gan import AttnGAN
from src.generator import Generator
from src.discriminator import Discriminator, PatchDiscriminator

@click.command()
@click.argument('epochs')
@click.argument('name')
@click.option('--gan', default=None)
@click.option('--damsm', default=None)
@click.option('--device', default=DEVICE)
def train_gan(epochs, name, gan, damsm, device):
    cub = CUB()
    if damsm is not None:
        damsm_model = DAMSM.load(damsm)
    if gan is None:
        generator = Generator(device)
        discriminator = Discriminator(device)
        # discriminator = PatchDiscriminator(device)
        gan = AttnGAN(damsm_model, generator, discriminator, device)
    metrics = gan.train(cub, epochs, fid_evaluator=FIDEvaluator)
    gan.save(name, metrics=metrics)
    return gan, metrics


if __name__ == '__main__':
    train_gan()
