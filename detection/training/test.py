import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from Unet import Unet
from dataset import DirDataset


def predict(net, img, device='cpu', threshold=0.5):
    ds = DirDataset('', '')
    _img = (ds.preprocess(img))

    _img = _img.unsqueeze(0)
    _img = _img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        o = net(_img)

        _o = o[:, 5, :, :]
        o = o[:, :5,:,:]

        if net.n_classes > 1:
            probs = torch.nn.functional.softmax(o, dim=1)
        else:
            probs = torch.sigmoid(o)

        probs = probs.squeeze(0)
        probs = probs.cpu()
        mask = probs.squeeze().cpu().numpy()

        _probs = torch.sigmoid(_o)
        _probs = _probs.squeeze(0)
        _probs = _probs.cpu()
        _mask = _probs.squeeze().cpu().numpy()
    return (mask, _mask > threshold )


def mask_to_image(mask):
    return Image.fromarray(( mask ).astype(np.uint8))


def main(hparams):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Unet.load_from_checkpoint(hparams.checkpoint)
    net.freeze()
    net.to(device)
    net.eval()
    for fn in tqdm(os.listdir(hparams.img_dir)):
        fp = os.path.join(hparams.img_dir, fn)

        img = Image.open(fp)
        mask, _mask = predict(net, img, device=device)
        mask = mask.argmax(axis = 0)

        mask_img = mask_to_image(mask)
        _mask_img = mask_to_image(_mask)
        mask_img.save(os.path.join(hparams.out_dir, fn[:-4] + '_2.png'))
        _mask_img.save(os.path.join(hparams.out_dir, fn[:-4] + '_1.png'))


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--checkpoint', required=True)
    parent_parser.add_argument('--img_dir', required=True)
    parent_parser.add_argument('--out_dir', required=True)

    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)
