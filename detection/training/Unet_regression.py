from argparse import ArgumentParser
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split

from pytorch_lightning import LightningModule

from dataset_regression import DirDataset

import torchvision



def l2(y, y_hat):
    return (y[0] - y_hat[0])*(y[0] - y_hat[0]) + (y[1] - y_hat[1])*(y[1]- y_hat[1])


class Unet(LightningModule):
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

        num_classes: int = 4
        self.hparams = hparams


        self.model = torchvision.models.resnet18(pretrained=False, progress=True, num_classes = num_classes)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y  = batch


        y_hat = self.forward(x)


        loss =       l2(y[0], y_hat[0])+\
                     l2(y[1], y_hat[1])+\
                     l2(y[2], y_hat[2])+\
                     l2(y[3], y_hat[3])



        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch

        y_hat = self.forward(x)

        loss = 0#self.loss_func(y_hat, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)

        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return [optimizer]

    def __dataloader(self):
        dataset = self.hparams.dataset
        dataset = DirDataset(f'{dataset}/img', f'{dataset}/mask')

        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val
        print(len(dataset))

        train_ds, val_ds = random_split(dataset, [n_train, n_val]) #, generator=torch.Generator().manual_seed(347))
        train_loader = DataLoader(train_ds, batch_size=self.hparams.batch_size,num_workers=32, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.hparams.batch_size,num_workers=32, pin_memory=True, shuffle=False)

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
