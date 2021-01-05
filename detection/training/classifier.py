
from argparse import ArgumentParser

import torch
import torch.nn.functional as F

from torch import nn, optim

from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

from dataset_classifier import DirDataset


from torchvision import transforms, datasets, models


import torchvision

class Resnet(pl.LightningModule):
    def __init__(self, hparams):
        super(Resnet, self).__init__()
        self.hparams = hparams
        self.model_ft = models.resnet18()
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(self.num_ftrs, 500)
        self.H = 10

        self.model_ft_2 = nn.Sequential(
                nn.Linear(500 + self.H * self.H,224),
                nn.BatchNorm1d(224),
                nn.ReLU(),
                nn.Linear(224, self.H * self.H)
        )

    def forward(self, input):
        img, bytecode = input
        hidden = self.model_ft(img)
        hidden = hidden.reshape(-1, 500)

        concated_hidden = torch.cat((nn.Sigmoid()(hidden), bytecode.reshape(-1, self.H * self.H)), dim = 1)

        return self.model_ft_2(concated_hidden.reshape(-1,500 + self.H * self.H)).reshape(-1,  self.H, self.H)

    def training_step(self, batch, batch_nb):
        x, y = batch

        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch

        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)


        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=8e-3)
        return [optimizer]

    def __dataloader(self):
        dataset = self.hparams.dataset
        dataset = DirDataset(f'{dataset}/ssimg', f'{dataset}/simg')

        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val

        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(347))
        train_loader = DataLoader(train_ds, batch_size=16,num_workers=12, pin_memory=True, shuffle=True,drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=16,num_workers=12, pin_memory=True, shuffle=False,drop_last=True)

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
