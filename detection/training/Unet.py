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


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
    )
class Unet(pl.LightningModule):
    def __init__(self, hparams):
        super(Unet, self).__init__()
        self.hparams = hparams
#
        self.n_channels = 3#hparams.n_channels
        self.n_classes = 6#hparams.n_classes
        self.bilinear = True
        self.cross_entropy_weights =torch.tensor([1,100, 100, 100, 100]).float().cuda()
#
        base_model = models.resnet18(pretrained=True)
#
        base_layers = list(base_model.children())
#
        self.layer0 = nn.Sequential(*base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512,1, 0)
#
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
#
        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
#
        self.conv_last = nn.Conv2d(64, self.n_classes, 1)
    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
#
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
#
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
#
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
#
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)
#
        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)
#
        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
#
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)
#
        out = self.conv_last(x)
#
        return out


# import torchvision
# resnet = torchvision.models.resnet.resnet50(pretrained=True)


# class ConvBlock(nn.Module):
#     """
#     Helper module that consists of a Conv -> BN -> ReLU
#     """

#     def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.LeakyReLU()
#         self.with_nonlinearity = with_nonlinearity

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         if self.with_nonlinearity:
#             x = self.relu(x)
#         return x


# class Bridge(nn.Module):
#     """
#     This is the middle layer of the UNet which just consists of some
#     """

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.bridge = nn.Sequential(
#             ConvBlock(in_channels, out_channels),
#             ConvBlock(out_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.bridge(x)


# class UpBlockForUNetWithResNet50(nn.Module):
#     """
#     Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
#     """

#     def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
#                  upsampling_method="bilinear"):
#         super().__init__()

#         if up_conv_in_channels == None:
#             up_conv_in_channels = in_channels
#         if up_conv_out_channels == None:
#             up_conv_out_channels = out_channels

#         if upsampling_method == "conv_transpose":
#             self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
#         elif upsampling_method == "bilinear":
#             self.upsample = nn.Sequential(
#                 nn.Upsample(mode='bilinear', scale_factor=2, align_corners =True),
#                 nn.Conv2d(up_conv_in_channels,up_conv_out_channels, kernel_size=1, stride=1)
#             )
#         self.conv_block_1 = ConvBlock(in_channels, out_channels)
#         self.conv_block_2 = ConvBlock(out_channels, out_channels)

#     def forward(self, up_x, down_x):
#         """
#         :param up_x: this is the output from the previous up block
#         :param down_x: this is the output from the down block
#         :return: upsampled feature map
#         """
#         x = self.upsample(up_x)
#         x = torch.cat([x, down_x], 1)
#         x = self.conv_block_1(x)
#         x = self.conv_block_2(x)
#         return x


# class Unet(pl.LightningModule):
#     DEPTH = 6

#     def __init__(self, hparams):
#         super().__init__()
#         self.hparams = hparams
#         self.cross_entropy_weights =torch.tensor([1,100, 100, 100, 100]).float().cuda()
#         resnet = torchvision.models.resnet.resnet50(pretrained=True)
#         down_blocks = []
#         up_blocks = []
#         self.input_block = nn.Sequential(*list(resnet.children()))[:3]
#         self.input_pool = list(resnet.children())[3]
#         for bottleneck in list(resnet.children()):
#             if isinstance(bottleneck, nn.Sequential):
#                 down_blocks.append(bottleneck)
#         self.down_blocks = nn.ModuleList(down_blocks)
#         for parameter in self.down_blocks:
#             parameter.requires_grad = False;
#     # for param in self.down_blocks.parameters():
#         #     param.requires_grad = False
#         #self.bridge = Bridge(2048, 2048)
#         up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
#         up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
#         up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
#         up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128  + 64, out_channels=128,
#                                                     up_conv_in_channels=256, up_conv_out_channels=128))
#         up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3  , out_channels=64 ,
#                                                     up_conv_in_channels=128, up_conv_out_channels=64))

#         self.up_blocks = nn.ModuleList(up_blocks)

#         self.out = nn.Conv2d(64, 6, kernel_size=1, stride=1)

#     def forward(self, x, with_output_feature_map=False):
#         pre_pools = dict()
#         pre_pools[f"layer_0"] = x
#         x = self.input_block(x)
#         pre_pools[f"layer_1"] = x
#         x = self.input_pool(x)

#         for i, block in enumerate(self.down_blocks, 2):
#             x = block(x)
#             if i == (Unet.DEPTH - 1):
#                 continue
#             pre_pools[f"layer_{i}"] = x

#         #x = self.bridge(x)

#         for i, block in enumerate(self.up_blocks, 1):
#             key = f"layer_{Unet.DEPTH - 1 - i}"
#             x = block(x, pre_pools[key])
#         x = self.out(x)

#         return x


    def training_step(self, batch, batch_nb):
        x, y = batch


        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat[:,:5,:,:], y[:,1,:,:].squeeze(1).long(), weight = self.cross_entropy_weights)
        loss += F.binary_cross_entropy_with_logits(y_hat[:, 5], y[:,0])

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch

        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat[:,:5,:,:], y[:,1,:,:].squeeze(1).long(), weight = self.cross_entropy_weights)
        loss += F.binary_cross_entropy_with_logits(y_hat[:, 5], y[:,0])

        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-5)

    def __dataloader(self):
        dataset = self.hparams.dataset
        dataset = DirDataset(f'./dataset/{dataset}/img', f'./dataset/{dataset}/mask')

        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val

        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=1,num_workers=4, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1,num_workers=4, pin_memory=True, shuffle=False)

        return {
            'train': train_loader,
            'val': val_loader,
        }

    @pl.data_loader
    def train_dataloader(self):
        return self.__dataloader()['train']

    @pl.data_loader
    def val_dataloader(self):
        return self.__dataloader()['val']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--n_channels', type=int, default=3)
        parser.add_argument('--n_classes', type=int, default=6)
        return parser
