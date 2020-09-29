import os
from argparse import ArgumentParser

import numpy as np
import torch

from classifier import Resnet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateLogger
from pytorch_lightning.profiler import AdvancedProfiler



def main(hparams):
    print(hparams.dataset)
    model = Resnet(hparams)
    if hparams.checkpoint != None:
        model = Resnet.load_from_checkpoint(hparams.checkpoint)
    model.train()

    os.makedirs(hparams.log_dir, exist_ok=True)
    try:
        log_dir = sorted(os.listdir(hparams.log_dir))[-1]
    except IndexError:
        log_dir = os.path.join(hparams.log_dir, 'version_0')

    checkpoint_callback = ModelCheckpoint(
        # monitor = 'loss',
        filepath=os.path.join(log_dir, 'checkpoints_older'),
        save_top_k=1,
        verbose=True,
    )
    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=60000,
        verbose=True,

    )

    lr_logger = LearningRateLogger()

    trainer = Trainer(
        gpus=1,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=stop_callback,
        callbacks= [lr_logger],
        accumulate_grad_batches=1,
        # resume_from_checkpoint=hparams.checkpoint,
        benchmark=True,
        # overfit_batches=10,
        # val_check_interval=0.250,
        # auto_scale_batch_size='binsearch',
        #gradient_clip_val=100,
        #amp_level='O2',
        #precision=16,
    )



    trainer.fit(model)


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', required=True)
    parent_parser.add_argument('--log_dir', default='lightning_logs')
    parent_parser.add_argument('--checkpoint', default=None)
    parent_parser.add_argument('--batch_size', type=int, default=1)
    parser = Resnet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)
