import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl

from dataset_classifier import DirDataset
from torch import nn
from torch.nn import functional as F
import torch

from torchvision import transforms, datasets, models


import torchvision
resnet = torchvision.models.resnet.resnet50(pretrained=True)

class Resnet(pl.LightningModule):
    def __init__(self, hparams):
        super(Resnet, self).__init__()
        self.hparams = hparams
        self.model_ft = models.resnet18(pretrained=True)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(self.num_ftrs, 5)

    def forward(self, input):
        return self.model_ft(input)

    def training_step(self, batch, batch_nb):
        x, y = batch


        y_hat = self.forward(x)
        y=y.squeeze(1)
        loss = F.cross_entropy(y_hat, y.long())

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch

        y_hat = self.forward(x)
        y=y.squeeze(1)
        loss = F.cross_entropy(y_hat, y.long())


        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.3, patience = 3)
        return [optimizer] , [scheduler]

    def __dataloader(self):
        dataset = self.hparams.dataset
        dataset = DirDataset(f'./dataset/{dataset}/simg')

        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val

        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(347))
        train_loader = DataLoader(train_ds, batch_size=8,num_workers=4, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=8,num_workers=4, pin_memory=True, shuffle=False)

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
