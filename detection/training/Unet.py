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

from dataset import DirDataset
from torch import nn
from torch.nn import functional as F
import torch

from torchvision import transforms, datasets, models


import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F





class DoubleConv(nn.Module):
    """
    Double Convolution and BN and ReLU
    (3x3 conv -> BN -> ReLU) ** 2
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Combination of MaxPool2d and DoubleConv in series
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path,
    followed by double 3x3 convolution.
    """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Unet(pl.LightningModule):
    """
    Architecture based on U-Net: Convolutional Networks for Biomedical Image Segmentation
    Link - https://arxiv.org/abs/1505.04597
    Parameters:
        num_classes: Number of output classes required (default 19 for KITTI dataset)
        num_layers: Number of layers in each side of U-net
        features_start: Number of features in first layer
        bilinear: Whether to use bilinear interpolation or transposed
            convolutions for upsampling.
    """

    def __init__(
            self,
            hparams,
    ):
        super().__init__()

        num_classes: int = 2
        num_layers: int = 4
        features_start: int = 64
        bilinear: bool = True

        self.hparams = hparams


        self.num_layers = num_layers

        layers = [DoubleConv(3, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2


        layers.append(nn.Conv2d(feats, int(num_classes), kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])
    def training_step(self, batch, batch_nb):
        x, y = batch


        y_hat = self.forward(x)


        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch

        y_hat = self.forward(x)

        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # avg_corner_loss = torch.stack([x['val_corner_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
<<<<<<< HEAD
        optimizer = torch.optim.Adam(self.parameters(), lr=16*10*4e-4)
=======
        optimizer = torch.optim.Adam(self.parameters(), lr=16*4*10*4e-4)
>>>>>>> 820c9a0f2f21e576c706800edf6b9a1c8ea9c104
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.3, patience = 3)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1.0, steps_per_epoch=40, epochs=10)
        return [optimizer]

    def __dataloader(self):
        dataset = self.hparams.dataset
        dataset = DirDataset(f'{dataset}/img', f'{dataset}/mask')

        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val
        print(len(dataset))

        train_ds, val_ds = random_split(dataset, [n_train, n_val]) #, generator=torch.Generator().manual_seed(347))
        train_loader = DataLoader(train_ds, batch_size=self.hparams.batch_size,num_workers=8, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.hparams.batch_size,num_workers=8, pin_memory=True, shuffle=False)

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
