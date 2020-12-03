import os, glob, cv2
import random
from tqdm import tqdm
from PIL import Image
import numpy as np, torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from torchvision import transforms, datasets, models

class DirDataset(Dataset):

    def __init__(self, img_dir, scale=1):
        self.img_dir = img_dir
        self.scale = scale
        print(self.img_dir)
        try:
            self.ids = (sorted([os.path.splitext(s)[0] for s in os.listdir(self.img_dir)]))
            self.ids.reverse()

        except FileNotFoundError:
            self.ids = []

    def __len__(self):
        return min(3000000, len(self.ids))

    def preprocess(self, img, mask=False):

        if not mask:
            trans = transforms.Compose([
                # transforms.RandomGrayscale(),

                # transforms.ColorJitter(0.3,0.3, 0.3, 0.1 ),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.5, scale=(0.01, 0.1), ratio=(0.3, 3.3), value=(1,0,1), inplace=True),
                # transforms.RandomErasing(p=0.5, scale=(0.01, 0.1), ratio=(0.3, 3.3), value=(1,0,1), inplace=True),
                # transforms.RandomErasing(p=0.5, scale=(0.01, 0.1), ratio=(0.3, 3.3), value=(1,0,1), inplace=True),
                # transforms.RandomErasing(p=0.5, scale=(0.01, 0.1), ratio=(0.3, 3.3), value=(1,0,1), inplace=True),
                # transforms.RandomErasing(p=0.5, scale=(0.01, 0.1), ratio=(0.3, 3.3), value=(1,0,1), inplace=True),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])


            img = trans(img).float()

        return img

    def __getitem__(self, i):
        idx = self.ids[i ]
        img_files = [os.path.join(self.img_dir, idx + '.jpg')]
        label_files = [os.path.join(self.img_dir, idx + '.txt')]


        # assert len(img_files) == 1, f"{idx}: {img_files}"
        # assert len(mask_files) == 2, f"{idx}: {mask_files}"

        img = Image.open(img_files[0])
        if os.path.exists(label_files[0]):

            file = open(label_files[0],"r")
            label = int(file.read())
        else:
            label = 587
            trans = transforms.Compose([
                transforms.RandomCrop(size = (224, 224), pad_if_needed=True)
                ])

            img = trans(img)



        if random.random() > 0.5:

            img = transforms.functional.hflip(img)

        # Random vertical flipping
        if random.random() > 0.5:
            img = transforms.functional.vflip(img)


        # Random affine
        if random.random() > 0.3:
            rotation = random.randint(-180, 180)
            translate = [random.randint(-10,10), random.randint(-10,10)]
            scale = 1.0
            shear = random.randint(-50, 50)

            img = transforms.functional.affine(img, rotation, translate, scale, shear)



        img = self.preprocess(img)
        # cv2.imshow("a",img.cpu().numpy().transpose(1, 2, 0))
        # cv2.waitKey(0);



        return (
            img,
            torch.ones((1,))*label)
