from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
import numpy as np
import os.path as osp
import io
import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import *
from torchvision.transforms import functional as F
def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDatasettest(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        # print(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        return img, pid, camid







class ImageDatasettrain(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, height, weight):
        self.dataset = dataset
        self.height=height
        self.weight=weight
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img1 = read_image(img_path)
        aa=img_path.split('/')
        aa[1]='market1501_ori'
        path2='/'.join(aa)
        img2=read_image(path2)
        

        width, height = img1.size
        resolution=(width*1.0)/self.width
        Random2DT=Random2DTranslation(self.height,self.width)
        RandomHor=RandomHorizontalFlip2()
        toten=ToTensor()
        normm=Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


        img1,img2=Random2DT.twoimages(img1,img2)
        img1,img2=RandomHor(img1,img2)
        img1,img2=toten(img1),toten(img2)
        img1,img2=normm(img1),normm(img2)
        return img1, pid, camid,img2,resolution







class RandomHorizontalFlip2(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1,img2):

        if random.random() < self.p:
            return F.hflip(img1),F.hflip(img2)
        return img1,img2




class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
    - height (int): target height.
    - width (int): target width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
        - img (PIL Image): Image to be cropped.
        """
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)
        
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img

    def twoimages(self, img1,img2):
        """
        Args:
        - img (PIL Image): Image to be cropped.
        """
        if random.uniform(0, 1) > self.p:
            return img1.resize((self.width, self.height), self.interpolation),img2.resize((self.width, self.height), self.interpolation)
        
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img1 = img1.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img1 = resized_img1.crop((x1, y1, x1 + self.width, y1 + self.height))
        resized_img2 = img2.resize((new_width, new_height), self.interpolation)
        croped_img2 = resized_img2.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img1, croped_img2