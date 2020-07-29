import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms

from Unet import Unet
from corners_to_crop import crop_to_corners
from dataset import DirDataset

from torch import nn
from torchvision import datasets, models, transforms

def predict(net, img, device='cpu', threshold=0.05):
    ds = DirDataset('', '')
    _img = (ds.preprocess(img))

    _img = _img.unsqueeze(0)
    _img = _img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        o = net(_img)

        _o = o[:, 5, :, :]
        o = o[:, :5,:,:]

        probs = torch.nn.functional.softmax(o, dim=1)
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

    #Load the identification network
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 587)

    identification_net = model_ft.to(device)
    identification_net.load_state_dict(torch.load(hparams.id_net))
    identification_net.eval()



    img = Image.open(hparams.img)
    img = img.resize((256, 256))
    mask, _mask = predict(net, img, device=device)
    mask = mask.argmax(axis = 0)

    img = np.array(img)

    print(img.dtype)

    mask = mask.astype(np.uint8)
    _mask = _mask.astype(np.uint8)
    crop_to_corners(identification_net, img, [_mask, mask], device)



if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help = False)
    parent_parser.add_argument('--checkpoint', required=True, help = "Network for segmentation")
    parent_parser.add_argument('--id_net', required=True, help = "Network for tag classification")
    parent_parser.add_argument('--img', required=True)

    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)
