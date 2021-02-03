
from argparse import ArgumentParser

import torch
import torch.nn.functional as F

from torch import nn, optim
import copy
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

from dataset_classifier import DirDataset


from torchvision import transforms, datasets, models


import torchvision

def loss_bit(input, target):
    return ((nn.Sigmoid()(input)>0.5).long()!=target.long()).reshape(16, -1).sum()
lis = []


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



class Resnet(pl.LightningModule):
    def __init__(self, hparams):
        super(Resnet, self).__init__()
        self.hparams = hparams
        self.model_ft = models.resnet50()
        self.num_ftrs = self.model_ft.fc.in_features


        self.loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([10.0]))
        self.H = 24

        self.model_ft.fc = nn.Linear(self.num_ftrs, self.H * self.H)

        self.model_ft_2 = nn.Sequential(
                nn.Linear(500, self.H * self.H),
        )

    def forward(self, input):
        img, bytecode = input
        # hidden = self.model_ft(img)
        # hidden = hidden.reshape(-1, 500)

        # concated_hidden = torch.cat((nn.Sigmoid()(hidden), bytecode.reshape(-1, self.H * self.H)), dim = 1)

        hidden = self.model_ft(img)
        hidden = hidden.reshape(-1, self.H, self.H)


        return hidden

    def training_step(self, batch, batch_nb):
        x, y = batch

        y_hat = self.forward(x)

        loss = self.loss_func(y_hat, y)

        bb = loss_bit(y_hat,y)
        # with open("out.csv", "a") as f:
        #     for x in bb:
        #         f.write(str(x.item()) + '\n')



        self.log('bit', bb, on_step=True, on_epoch=False, prog_bar=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch

        y_hat = self.forward(x)
        loss= self.loss_func(y_hat, y)
        # print("loss_bit", loss_bit(y_hat, y))


        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=8e-4)
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
            'train': train_loader, 'val': val_loader, }

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
