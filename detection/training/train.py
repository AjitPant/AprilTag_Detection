import os
from argparse import ArgumentParser

import numpy as np
import torch

from Unet import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import AdvancedProfiler



def main(hparams):
    print(hparams.dataset)
    model = Unet(hparams)

    model.train()

    os.makedirs(hparams.log_dir, exist_ok=True)
    try:
        log_dir = sorted(os.listdir(hparams.log_dir))[-1]
    except IndexError:
        log_dir = os.path.join(hparams.log_dir, 'version_0')

    checkpoint_callback = ModelCheckpoint(
        monitor = 'loss',
        filepath=os.path.join(log_dir, 'checkpoints'),
        save_top_k=4,
        verbose=True,
    )
    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=60000,
        verbose=True,

    )


    trainer = Trainer(
        num_nodes=1,
        accelerator='ddp',
        gpus=hparams.n_gpu,
        checkpoint_callback=checkpoint_callback,
        accumulate_grad_batches=1,
        auto_scale_batch_size='binsearch',
        # resume_from_checkpoint=hparams.checkpoint,
        benchmark=True,
    )



    trainer.fit(model)


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', required=True)
    parent_parser.add_argument('--n_gpu', default = 1, type = int)
    parent_parser.add_argument('--log_dir', default='/raid/apant_ma/AprilTag-Detection/AprilTag_Detection/detection/training/lightning_logs')
    parent_parser.add_argument('--checkpoint', default=None)
    parent_parser.add_argument('--batch_size', type=int, default=1)
    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)
