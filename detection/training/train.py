import os
#######################import shutil
from argparse import ArgumentParser

import numpy as np
import torch

from Unet import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import AdvancedProfiler



def main(hparams):
###############3i####    shutil.rmtree('/raid/apant_ma/AprilTag-Detection/AprilTag_Detection/detection/training/lightning_logs')
    print(hparams.dataset)
    model = Unet(hparams)

    model.train()

    os.makedirs(hparams.log_dir, exist_ok=True)
    log_dir = os.path.join(hparams.log_dir, 'version_4')

    assert hparams.checkpoint is None or  os.path.exists(hparams.checkpoint)

    checkpoint_callback = ModelCheckpoint(
        monitor = 'loss',
        filepath=os.path.join(log_dir, 'checkpoints'),
        save_top_k=-1,
        verbose=True,
    )
    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=3,
        verbose=True,

    )


    trainer = Trainer(
        num_nodes=1,
        max_epochs = 10,
        accelerator='ddp',
        gpus=hparams.n_gpu,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=hparams.checkpoint,
#        benchmark=True,
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
