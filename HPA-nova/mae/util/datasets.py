# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms
from skimage.io import imread
from pathlib import Path
import torch
import cv2
import numpy as np
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip, RandomAffine, RandomVerticalFlip, RandomPerspective, ColorJitter, RandomRotation)
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
        
def a_ordinary_collect_method(batch):
    '''
    I am a collect method for User Dataset
    '''
    img, exp = [], []

    study_id = []
    weight = []
    if len(batch[0]) == 2:
        for i, e in batch:
            img.append(i)
            exp.append(e)
        # return (torch.cat(img), torch.tensor(np.concatenate(pe)).long(),
                # torch.tensor(np.concatenate(exp)).float(), torch.tensor(np.concatenate(msk)).float(), cnt[0])
        return (torch.cat(img), torch.tensor(np.concatenate(exp)).float())

def custom_collate_fn(batch):
    # print("custom collating...")

    # OG, worked up to 1 epoch
    # ret = [
    #     torch.stack([x[0] for x in batch]), # samples
    #     torch.tensor([x[1] for x in batch]) # targets
    # ]

    ret = [
        torch.stack([x[0] for x in batch]), # samples
        torch.tensor(np.array([x[1] for x in batch])) # targets
    ]

    # print(ret)
    return ret


class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        
        train = pd.read_csv(csv_file)
        train_meta = train[:]
        self.df = train_meta.reset_index(drop=True)
        self.cols = ['class{}'.format(i) for i in range(19)]
                                                            
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            RandomRotation(degrees=(0, 180)),
            RandomVerticalFlip(p=0.5),
            RandomHorizontalFlip(p=0.5),
            RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            # RandomPerspective(distortion_scale=0.5, p=0.5),
            AddGaussianNoise(mean=0.0, std=0.1),
            Normalize(mean=[0.0979, 0.06449, 0.062307, 0.098419], std=[0.14823, 0.0993746, 0.161757, 0.144149]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        cnt = 10 # 10 cells
        label = np.zeros((cnt, 19)) 
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        img = torch.zeros((cnt, 3, 224, 224))
        
        for i in range(cnt):
            cell_path = f'{img_path}_cell{i+1}.png'
            img_cell = imread(cell_path)
            img_cell = cv2.resize(img_cell, (224,224))
            img_cell = self.transform(img_cell)
            
            # Extracting the R and Y channels --> Merge
            r_channel = img_cell[0]  # Red channel
            y_channel = img_cell[3]  # Yellow channel
            
            # Merging R and Y channels
            ry_channel = r_channel + y_channel
            
            # Creating the new image tensor with merged R+Y channel
            img_cell = torch.stack([ry_channel, img_cell[1], img_cell[2]])
            img[i] = img_cell
            
            # img_cell = self.tensor_tfms(torch.tensor(img_cell).double())
            label_cell = row[self.cols].values.astype(np.float64)
            label[i] = torch.tensor(label_cell.reshape(1, -1))
            
            # print(f'img: {img_cell.shape}, label: {label.shape}') # img: torch.Size([4, 228, 98]), label: (1, 19)
    
        return img_cell, 0.9*label_cell + 0.1/19


# custom, 4/2
class CustomImageDataset_old(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        img_path = f'{img_path}_cell1.png' # NEED TO TAKE IN ALL CELLS
        image = read_image(img_path)
        label = self.annotations.iloc[idx, 1]
        # print(f'img: {image.shape}, label: {label.shape}')
        return image, label


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    # root = os.path.join(args.data_path, 'train' if is_train else 'val')
    root = os.path.join(os.getcwd(), args.data_path[1:])
    # root = os.path.join(args.data_path, '') # , 'train' if is_train else 'val')
    # dataset = datasets.ImageFolder(root, transform=transform) # error, bc https://stackoverflow.com/questions/69199273/torchvision-imagefolder-could-not-find-any-class-folder
    if(is_train):
        csv_file = '../dataloaders/split/HPA_final_train.csv' # cwd: mae
    else:
        csv_file = '../dataloaders/split/HPA_final_valid.csv'
    dataset = CustomImageDataset(csv_file, root, transform=transform)
    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
