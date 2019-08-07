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
        loader_config = {
            'batch_size': batch_size,
            'shuffle': True,
            'drop_last': True,
            'collate_fn': dataset.collate_fn
        }
        train_loader = DataLoader(dataset.train, **loader_config)
        test_loader = DataLoader(dataset.test, **loader_config)

        for e in tqdm(range(epoch), desc='Epochs'):
            self.gen.train(), self.disc.train()

            train_pbar = tqdm(train_loader, desc='Training', leave=False)
            for batch in train_pbar:
                self.gen.zero_grad(), self.disc.zero_grad()

    def batch_loss(self, batch):
        pass
