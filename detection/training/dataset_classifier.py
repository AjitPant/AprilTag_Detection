import os, glob
import random
import pickle

from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset

from torchvision import transforms

class DirDataset(Dataset):

    def __init__(self, img_dir, label_dir ,scale=1):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.scale = scale
        print(self.img_dir)
        try:
            self.ids = (sorted([os.path.splitext(s)[0] for s in os.listdir(self.img_dir)]))
            self.ids.reverse()

        except FileNotFoundError:
            self.ids = []

    def __len__(self):
        return min(30000, len(self.ids))

    def preprocess(self, img, mask=False):

        if not mask:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])


            img = trans(img).float()

        return img

    def __getitem__(self, i):
        idx = self.ids[i ]
        img_files = os.path.join(self.img_dir, idx + '.jpg')
        label_files = os.path.join(self.label_dir, idx + '.pkl')



        img = Image.open(img_files)


        img = self.preprocess(img)

        with open(label_files, "rb") as f:
            label = pickle.load(f)

        return (
            img,
            torch.tensor(label)
            )
