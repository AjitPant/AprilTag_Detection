
from argparse import ArgumentParser

import torch
import torch.nn.functional as F

from torch import nn, optim

from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

from dataset_classifier import DirDataset



class Resnet(pl.LightningModule):
    def __init__(self, hparams):
        super(Resnet, self).__init__()
        self.hparams = hparams
        self.model_ft = nn.Sequential(
                        nn.Linear(3*224*224, 2*224),
                        nn.ReLU(),
                        nn.Linear(2*224, 2*224),
                        nn.ReLU(),
                        nn.Linear(2*224, 224),
                        nn.ReLU(),
                        nn.Linear(224, 140),
                        nn.ReLU(),
                        nn.Linear(140, 100)
        )

    def forward(self, input):
        return self.model_ft(input.reshape(-1,3*224*224))

    def training_step(self, batch, batch_nb):
        x, y = batch


        y_hat = self.forward(x)
        y=y.squeeze(1)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch

        y_hat = self.forward(x)
        y=y.squeeze(1)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)


        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return [optimizer]

    def __dataloader(self):
        dataset = self.hparams.dataset
        dataset = DirDataset(f'{dataset}/ssimg', f'{dataset}/simg')

        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val

        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(347))
        train_loader = DataLoader(train_ds, batch_size=16,num_workers=32, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16,num_workers=32, pin_memory=True, shuffle=False)

        return {
            'train': train_loader,
            'val': val_loader,
        }

    def train_dataloader(self):
        return self.__dataloader()['train']

    def val_dataloader(self):
        return self.__dataloader()['val']


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--n_channels', type=int, default=3)
        parser.add_argument('--n_classes', type=int, default=6)
        return parser
