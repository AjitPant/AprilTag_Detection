import os
from argparse import ArgumentParser

import numpy as np
import torch

from classifier import Resnet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import AdvancedProfiler



def main(hparams):
    print(hparams.dataset)
    model = Resnet(hparams)
    if hparams.checkpoint != None:
        model = Resnet.load_from_checkpoint(hparams.checkpoint)
    model.train()

    os.makedirs(hparams.log_dir, exist_ok=True)
    log_dir = os.path.join(hparams.log_dir, 'version_0')

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(log_dir, 'checkpoints_older'),
        save_top_k=-1,
        verbose=True,
    )
    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=60000,
        verbose=True,

    )


    trainer = Trainer(
        gpus=1,
        accelerator = 'ddp',
        checkpoint_callback=checkpoint_callback,
        accumulate_grad_batches=1,
        resume_from_checkpoint=hparams.checkpoint,
        benchmark=True,
    #    default_root_dir='/raid/apant_ma/AprilTag-Detection/AprilTag_Detection/detection/training/lightning_logs',

    )



    trainer.fit(model)


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', required=True)
    parent_parser.add_argument('--log_dir', default='/raid/apant_ma/AprilTag-Detection/AprilTag_Detection/detection/training/lightning_logs')
    parent_parser.add_argument('--checkpoint', default=None)
    parent_parser.add_argument('--batch_size', type=int, default=1)
    parser = Resnet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)
