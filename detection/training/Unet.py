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


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0


    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()



class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        # print(self.focal(input, target))
        # print(torch.log(dice_loss(input, target)))
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()


import torchvision
resnet = torchvision.models.resnet.resnet50(pretrained=True)


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="bilinear"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners =True),
                nn.Conv2d(up_conv_in_channels,up_conv_out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class Unet(pl.LightningModule):
    DEPTH = 6

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.loss_func = MixedLoss(10.0, 2.0);
        self.cross_entropy_weights =torch.tensor([1,100, 100, 100, 100],dtype = torch.float32, device = torch.device('cuda:0'))
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        #self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128  + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3  , out_channels=64 ,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out = nn.Conv2d(64, 6, kernel_size=1, stride=1)
    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (Unet.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        #x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{Unet.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        x = self.out(x)

        return x


    def training_step(self, batch, batch_nb):
        x, y = batch


        y_hat = self.forward(x)
        # loss = F.cross_entropy(y_hat[:,:5], y[:,1].squeeze(1).long(), weight = self.cross_entropy_weights)

        loss = torch.zeros(1)

        for batch in range(y_hat.shape[0]):
            for i in range(5):

                loss += self.loss_func(y_hat[batch,i], y[batch,1] == i)
        corner_loss = loss.clone()
        # loss += F.binary_cross_entropy_with_logits(y_hat[:, 5], y[:,0])
        for batch in range(y_hat.shape[0]):
            loss += self.loss_func(y_hat[batch,5], y[batch,0])

        tensorboard_logs = {'train_loss': loss, 'train_corner_loss' : corner_loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch

        y_hat = self.forward(x)

        # loss = F.cross_entropy(y_hat[:,:5], y[:,1].squeeze(1).long(), weight = self.cross_entropy_weights)
        loss = torch.zeros(1)
        for batch in range(y_hat.shape[0]):
            for i in range(5):

                loss += self.loss_func(y_hat[batch,i], y[batch,1] == i)
        corner_loss = loss.clone()
        for batch in range(y_hat.shape[0]):
            loss += self.loss_func(y_hat[batch,5], y[batch,0])
        # loss += F.binary_cross_entropy_with_logits(y_hat[:, 5], y[:,0])

        return {'val_loss': loss, 'val_corner_loss' : corner_loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_corner_loss = torch.stack([x['val_corner_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_corner_loss': avg_corner_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.3, patience = 3)
        return [optimizer] , [scheduler]

    def __dataloader(self):
        dataset = self.hparams.dataset
        dataset = DirDataset(f'./dataset/{dataset}/img', f'./dataset/{dataset}/mask')

        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val

        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(47))
        train_loader = DataLoader(train_ds, batch_size=4,num_workers=4, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4,num_workers=4, pin_memory=True, shuffle=False)

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
