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


from torch.utils.data import DataLoader

def predict(net, img, device='cpu', threshold=0.5):
    _img = img
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
        mask = probs.squeeze()#.cpu().numpy()

        _probs = torch.sigmoid(_o)
        _probs = _probs.squeeze(0)
        _probs = _probs.cpu()
        _mask = _probs.squeeze()#.cpu().numpy()
    return (mask, _mask > threshold )


def mask_to_image(mask):
    return Image.fromarray(( mask ).astype(np.uint8))

def iou(pred, target, n_classes = 5):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for background class ("0")
  for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu()  # Cast to long to prevent overflows
    print(intersection)
    union = pred_inds.long().sum().data.cpu() + target_inds.long().sum().data.cpu() - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection) / float(max(union, 1)))
  return np.array(ious)

def recall(pred, target, n_classes = 5):
    recalls = []
    pred = pred.view(-1)
    target = target.view(-1)


    # Ignore recall for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
      pred_inds = pred == cls
      target_inds = target == cls
      true_positive = (pred_inds[target_inds]).long().sum().data.cpu()  # Cast to long to prevent overflows
      n_positive = target_inds.long().sum().data.cpu()
      if n_positive == 0:
        recalls.append(float('nan'))  # If there is no ground truth, do not include in evaluation
      else:
        recalls.append(float(true_positive) / float(n_positive))
    return np.array(recalls)

def precision(pred, target, n_classes = 5):
    precisions = []
    pred = pred.view(-1)
    target = target.view(-1)


    # Ignore recall for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored

      pred_inds = pred == cls
      target_inds = target == cls
      true_positive = (pred_inds[target_inds]).long().sum().data.cpu()  # Cast to long to prevent overflows
      n_positive = pred_inds.long().sum().data.cpu()

      if n_positive == 0:
        precisions.append(float('nan'))  # If there is no ground truth, do not include in evaluation
      else:
        precisions.append(float(true_positive) / float(n_positive))
    return np.array(precisions)



def main(hparams):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # net = Unet.load_from_checkpoint(hparams.checkpoint)
    # net.freeze()
    # net.to(device)
    # net.eval()

    img_dir = os.path.join('./dataset', hparams.dataset, 'img')
    mask_dir = os.path.join('./dataset', hparams.dataset, 'mask')

    test_ds = DirDataset(img_dir, mask_dir)

    test_loader = DataLoader(test_ds, batch_size=1,num_workers=8, pin_memory=True, shuffle=False)

    # ious = []
    # precisions = []
    # recalls = []
    for step, (img, masks) in enumerate(test_loader):   # gives batch data

        if(masks[0,1,:,:].max()>=5):
            print('found image'+ str(step))
            cv2.imwrite("IasM.png", img.squeeze(1).cpu().numpy())
            cv2.imwrite("IsasM.png", masks[0,1,:,:].cpu().numpy())
            cv2.imwrite("IsadsM.png", masks[0,0,:,:].cpu().numpy())
        print(step)
        continue

        mask, _mask = predict(net, img, device=device)
        mask = mask.argmax(axis = 0)

        _iou =  iou( mask,  masks[0,1,:,:].long())
        _precision =  precision( mask,  masks[0,1,:,:].long())
        _recall =  recall( mask,  masks[0,1,:,:].long())

        print('IoU: ', _iou)
        print('Precision: ', _precision)
        print('Recall: ', _recall)

        ious.append(_iou)
        precisions.append(_precision)
        recalls.append(_recall)

    ious = np.array(ious)
    precisions = np.array(precisions)
    recalls = np.array(recalls)


    print('IoU: ', np.nanmean(ious, axis = 0))
    print('Precision: ', np.nanmean(precisions, axis = 0))
    print('Recall: ', np.nanmean(recalls, axis = 0))

        # mask_img = mask_to_image(mask)
        # _mask_img = mask_to_image(_mask)
        # mask_img.save(os.path.join(hparams.out_dir, fn[:-4] + '_2.png'))
        # _mask_img.save(os.path.join(hparams.out_dir, fn[:-4] + '_1.png'))


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--checkpoint', required=True)
    parent_parser.add_argument('--dataset',default = 'test')
    parent_parser.add_argument('--out_dir', default = 'out_dump')

    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)
