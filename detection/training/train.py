import os
from argparse import ArgumentParser

import numpy as np
import torch

from Unet import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def main(hparams):
    print(hparams.dataset)
    model = Unet(hparams)
    if(hparams.checkpoint):
       model = Unet.load_from_checkpoint(hparams.checkpoint)

    model.train()

    os.makedirs(hparams.log_dir, exist_ok=True)
    try:
        log_dir = sorted(os.listdir(hparams.log_dir))[-1]
    except IndexError:
        log_dir = os.path.join(hparams.log_dir, 'version_0')



    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(log_dir, 'checkpoints'),
        save_top_k=5,
        verbose=True,
    )
    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=50,
        verbose=True,
    )
    trainer = Trainer(
        gpus=1,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=stop_callback,
        #accumulate_grad_batches=4,
        #precision=16,
    )

    trainer.fit(model)


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', required=True)
    parent_parser.add_argument('--log_dir', default='lightning_logs')
    parent_parser.add_argument('--checkpoint', default ='')
    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)
