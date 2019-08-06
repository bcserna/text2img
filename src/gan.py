from torch.utils.data import DataLoader

from src.config import DEVICE, BATCH
from src.discriminator import Discriminator
from src.generator import Generator
from tqdm import tqdm


class AttnGAN(object):
    def __init__(self, text_encoder, device=DEVICE):
        self.gen = Generator().to(device)
        self.disc = Discriminator().to(device)
        self.text_encoder = text_encoder.to(device)
        self.device = device

    def train(self, dataset, epoch, batch=BATCH):
        train_loader = DataLoader(dataset.train, batch_size=batch, shuffle=True, drop_last=True,
                                  collate_fn=dataset.collate_fn)
        test_loader = DataLoader(dataset.test, batch_size=batch, shuffle=True, drop_last=True,
                                 collate_fn=dataset.collate_fn)


