import os, glob, cv2
import random
from tqdm import tqdm
from PIL import Image
import numpy as np, torch
from torch.utils.data import Dataset
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
        return min(30000, len(self.ids))

    def preprocess(self, img, mask=False):

        if not mask:
            trans = transforms.Compose([
                transforms.RandomGrayscale(),
                transforms.ColorJitter(0.3,0.3, 0.3, 0.1 ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.5, scale=(0.001, 0.01), ratio=(0.3, 3.3), value=(0,0,0), inplace=True),
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
        file = open(label_files[0],"r")

        label = int(file.read())

        if random.random() > 1.5:
            img = transforms.functional.hflip(img)

        # Random vertical flipping
        if random.random() > 1.5:
            img = transforms.functional.vflip(img)


        # Random affine
        if random.random() > 1.0:
            rotation = random.randint(-180, 180)
            translate = [random.randint(-10,10), random.randint(-10,10)]
            scale = 1.0
            shear = random.randint(-50, 50)

            img = transforms.functional.affine(img, rotation, translate, scale, shear)



        img = self.preprocess(img)



        return (
            img,
            torch.ones((1,))*label)
