import os, glob
import random
import pickle
import cv2

from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset

from torchvision import transforms

class DirDataset(Dataset):

    def __init__(self, img_dir, label_dir ,scale=1):
        self.img_dir = img_dir
        self.label_dir = label_dir
        #self.img_dir = './dataset/try/ssimg'
        #self.label_dir = './dataset/try/simg'
        self.scale = scale
        self.label_side = 224



        print(self.img_dir)
        try:
            self.ids = (sorted([os.path.splitext(s)[0] for s in os.listdir(self.img_dir)]))
            self.ids.reverse()

        except FileNotFoundError:
            self.ids = []

    def __len__(self):
        return len(self.ids)

    def preprocess(self, img, mask=False):

        if not mask:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])


            img = trans(img).float()

        return img


    def randomPad(self, label, bytecode):
        side_diff = self.label_side - label.shape[0]

        pad_width = (random.randint(0, side_diff), random.randint(0, side_diff))
        pad = ((pad_width[0], side_diff- pad_width[0]), (pad_width[1], side_diff-pad_width[1]))

        label = np.pad(label, pad, "constant", constant_values = 0)  # effectively zero padding
        bytecode = np.pad(bytecode, pad, "constant", constant_values = 3)  # effectively zero padding

        return label, bytecode




    def __getitem__(self, i):
        idx = self.ids[i ]
        img_files = os.path.join(self.img_dir, idx + '.jpg')
        label_files = os.path.join(self.label_dir, idx + '.pkl')



        img = Image.open(img_files)

        with open(label_files, "rb") as f:
            label, bytecode = pickle.load(f)
        label, bytecode = self.randomPad(label, bytecode)

        # with open("bytecode_36h11.pkl","wb") as f:
        #     pickle.dump(bytecode, f)
        # assert(False)
        # cv2.namedWindow("img"+str(idx), cv2.WINDOW_NORMAL)
        # cv2.namedWindow("label"+str(idx), cv2.WINDOW_NORMAL)
        # cv2.namedWindow("bytecode"+str(idx), cv2.WINDOW_NORMAL)

        # cv2.imshow("img"+str(idx), np.array(img))
        # cv2.imshow("label"+str(idx), np.array(label).reshape(15, 15))
        # cv2.imshow("bytecode"+str(idx), np.array(bytecode).astype(np.float32).reshape(15, 15))
        # cv2.waitKey(0)

        img = self.preprocess(img)

        label = torch.tensor(label/255)
        bytecode = torch.tensor(bytecode)



        return (
            (img, bytecode),
            label
            )
