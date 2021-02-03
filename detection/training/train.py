import os
from argparse import ArgumentParser

import numpy as np
import torch

from Unet import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import AdvancedProfiler

import torch.nn as nn



def main(hparams):
    print(hparams.dataset)
    model = Unet(hparams)
    # model = Unet.load_from_checkpoint(hparams.checkpoint,dataset = hparams.dataset, learning_rate = hparams.learning_rate)
#    model.loss_func2 = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([10.0]))

    model.train()

    os.makedirs(hparams.log_dir, exist_ok=True)
    log_dir = os.path.join(hparams.log_dir, 'version_4')

    assert hparams.checkpoint is None or  os.path.exists(hparams.checkpoint)

    checkpoint_callback = ModelCheckpoint(
        monitor ='wt' ,
        filepath=os.path.join(log_dir, 'checkpoints'),
        save_top_k=-1,
        verbose=True,
    )


    trainer = Trainer(
        num_nodes=1,
        max_epochs = 8000,
#        gradient_clip_val = 0.5,
        # accelerator=hparams.accelerator,
        gpus=hparams.n_gpu,
        checkpoint_callback=checkpoint_callback, \
       # resume_from_checkpoint=hparams.checkpoint,
#        precision = 16,
        sync_batchnorm = True,
        accumulate_grad_batches=1,
       benchmark=True,
        # default_root_dir='/raid/apant_ma/AprilTag-Detection/AprilTag_Detection/detection/training/lightning_logs',
    )




    trainer.fit(model)


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', required=True)
    parent_parser.add_argument('--n_gpu', default = 1, type = int)
    parent_parser.add_argument('--accelerator', default = 'ddp')
    parent_parser.add_argument('--log_dir', default='/raid/apant_ma/AprilTag-Detection/AprilTag_Detection/detection/training/lightning_log')
    parent_parser.add_argument('--checkpoint', default=None)
    parent_parser.add_argument('--batch_size', type=int, default=1)
    parent_parser.add_argument('--learning_rate', type=float, default=4*4e-4)
    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)
