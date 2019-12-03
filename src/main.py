import click

from src.config import DEVICE
from src.data import CUB
from src.damsm import DAMSM
from src.evaluation import IS_FID_Evaluator
from src.gan import AttnGAN


@click.group()
def main():
    pass


@main.command()
@click.argument('epochs', type=int)
@click.argument('name')
@click.option('--gan', default=None)
@click.option('--damsm', default=None)
@click.option('--device', default=DEVICE)
def train_gan(epochs, name, gan, damsm, device):
    cub = CUB()
    if damsm is not None:
        damsm_model = DAMSM.load(damsm, device=device)
    if gan is None:
        gan = AttnGAN(damsm_model, device)
    metrics = gan.train(cub, epochs, evaluator=IS_FID_Evaluator)
    gan.save(name, metrics=metrics)
    return gan, metrics


@main.command()
@click.argument('gan')
@click.argument('damsm')
@click.argument('save_dir')
@click.option('--device', default=DEVICE)
def validate_gan(gan, damsm, save_dir, device):
    dataset = CUB()
    damsm_model = DAMSM.load(damsm, device=device)
    gan_model = AttnGAN.load(gan, damsm_model, device=device)
    gan_model.validate_test_set(dataset, save_dir=save_dir)


@main.command()
@click.argument('epochs', type=int)
@click.argument('name')
@click.option('--patience', type=int, default=20)
@click.option('--device', default=DEVICE)
def train_damsm(epochs, name, patience, device):
    cub = CUB()
    damsm = DAMSM(len(cub.train.vocab), device=device)
    metrics = damsm.train(cub, epochs, patience=patience)
    damsm.save(name, metrics=metrics)
    return damsm, metrics


if __name__ == '__main__':
    main()
