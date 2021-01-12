from argparse import ArgumentParser
import copy
from deeplab import Res_Deeplab

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split

from pytorch_lightning import LightningModule

from dataset import DirDataset



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

def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1e-5

    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))








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

        num_classes: int = 2
        num_layers: int = 10
        features_start: int = 1
        bilinear: bool = False

        self.hparams = hparams
        self.wt = 1

        self.loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0]))
        self.loss_func2 = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([10.0]))
        #self.val_func = FocalLoss(alpha=2, gamma=2)
        # self.val_func = dice_loss
        self.dist_loss = nn.MSELoss()
        self.val_func = dice_loss

        self.model = Res_Deeplab(2)



    def forward(self, x):
        return self.model(x)



    def training_step(self, batch, batch_nb):
        x, y  = batch


        y_hat = self.forward(x)


        loss = self.loss_func((y_hat[:,1]), y[:,1])
       # dice2 = self.val_func(y_hat[:,0], y[:,0]) #+ self.val_func(y_hat[:,1], y[:,1]) + self.loss_func(y_hat, y)
        #dice = self.loss_func2(y_hat[:,0], y[:,0])

       # dist = self.dist_loss(nn.Sigmoid()(y_hat[:,0]), y[:,0])

        t_loss = (loss)


        self.log('bce', loss, on_step=True, on_epoch=False, prog_bar=True)
        #self.log('dice', dice, on_step=True, on_epoch=False, prog_bar=True)
        #self.log('dice2', dice2, on_step=True, on_epoch=False, prog_bar=True)
        #self.log('dist', dist, on_step=True, on_epoch=False, prog_bar=True)
        self.log('t_loss', t_loss, on_step=True, on_epoch=False, prog_bar=False)
        #self.log('wt', self.wt, on_step=False, on_epoch=True, prog_bar=False)


        return t_loss
    def training_epoch_end(self, training_step_outputs):
        self.wt *= 0.95
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
