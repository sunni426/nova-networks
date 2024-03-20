# nova-networks/HPA-nova
import numpy as np
import os
import pandas as pd
from skimage.io import imread
from pathlib import Path
import cv2

from typing import Callable, List
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip, RandomAffine, RandomVerticalFlip, RandomChoice, ColorJitter, RandomRotation)
import skimage
# from utils.tile_fix import tile
# from utils.ha import get_tiles
import random
from configs import Config
import tifffile
import torch
import math
import glob
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms


def a_ordinary_collect_method(batch):
    '''
    I am a collect method for User Dataset
    '''
    img, pe, exp, msk, cnt = [], [], [], [], []
    # debug
    study_id = []
    weight = []
    # debug end
    if len(batch[0]) == 5:
        for i, p, e, m, l in batch:
            img.append(i)
            pe.append(p)
            exp.append(e)
            msk.append(m)
            cnt.append(l)
        return (torch.cat(img), torch.tensor(np.concatenate(pe)).long(),
                torch.tensor(np.concatenate(exp)).float(), torch.tensor(np.concatenate(msk)).float(), cnt[0])

def normwidth(size, margin=32):
    outsize = size // margin * margin
    outsize = max(outsize, margin)
    return outsize


def resize_short(img, target_size):
    """ resize_short """
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(math.ceil(img.shape[1] * percent))
    resized_height = int(math.ceil(img.shape[0] * percent))

    # resized_width = normwidth(resized_width)
    # resized_height = normwidth(resized_height)
    resized = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4)
    return resized


class TrainDataset(Dataset):
    HEIGHT = 137
    WIDTH = 236

    def __init__(self, df: pd.DataFrame, images: pd.DataFrame,
                 image_transform: Callable, debug: bool = True, weighted_sample: bool = False,
                 square: bool = False):
        super().__init__()
        self._df = df
        self._images = images
        self._image_transform = image_transform
        self._debug = debug
        self._square = square
        # stats = ([0.0692], [0.2051])
        self._tensor_transform = Compose([
            ToTensor(),
            # Normalize(mean=[0.0692, 0.0692, 0.0692], std=[0.2051, 0.2051, 0.2051]),
             # Normalize(mean=[0.485, 0.456, 0.406,0.406], std=[0.229, 0.224, 0.225,0.225])
            Normalize(mean=[0.08323, 0.05766, 0.0554, 0.08365], std=[0.13553, 0.09504, 0.14648, 0.1332])
        ])
        if weighted_sample:
            # TODO: if weight sampler is necessary
            self.weight = self.get_weight()

    def get_weight(self):
        path = os.path.dirname(os.path.realpath(__file__)) + '/../../metadata/train_onehot.pkl'
        onehot = pd.read_pickle(path)
        exist = onehot.loc[self._df['id']]
        weight = []
        log = 1 / np.log2(exist.sum() + 32)
        for i in range(exist.shape[0]):
            weight.append((log * exist.iloc[i]).max())
        weight = np.array(weight)
        return weight

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int):
        item = self._df.iloc[idx]
        image = self._images[idx].copy()
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        # image = Image.fromarray(image)
        if self._image_transform:
            image = self._image_transform(image=image)['image']
        # else:
        image = self._tensor_transform(image)
        target = np.zeros(3)
        target[0] = item['grapheme_root']
        target[1] = item['vowel_diacritic']
        target[2] = item['consonant_diacritic']
        return image, target


class LandmarkDataset(Dataset):
    def __init__(self, df, tfms=None, size=256, tta=1, cfg: Config=None, test=False, scale=None, full=False):
        self.df = df
        self.tfms = tfms
        self.size = size
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.cfg = cfg
        self.scale = scale or []
        self.tensor_tfms = Compose([
            ToTensor(),
            #Normalize(mean=[0.485, 0.456, 0.406,0.406], std=[0.229, 0.224, 0.225,0.225])
            Normalize(mean=[0.08323, 0.05766, 0.0554, 0.08365], std=[0.13553, 0.09504, 0.14648, 0.1332]),
        ])
        self.tta = tta
        self.test = test
        self.full = full

    def __len__(self):
        return self.df.shape[0] * self.tta

    def __getitem__(self, idx: int):
        image_id = self.df.iloc[idx % self.df.shape[0]]['id']
        prefix = 'full' if self.full else 'clean_data'
        path = self.path / '../../../landmark/{}/train/{}/{}/{}/{}.jpg'.format(prefix,
            image_id[0], image_id[1], image_id[2], image_id
        )
        img = imread(path)[:, :, :3]
        # img = cv2.imread(path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))
        # resize short edge first
        # img = resize_short(img, self.cfg.transform.size)

        if self.tfms:
            return self.tensor_tfms(self.tfms(image=img)['image']), self.df.iloc[idx % self.df.shape[0]]['label']
        else:
            return self.tensor_tfms(img), self.df.iloc[idx % self.df.shape[0]]['label']


class STRDataset(Dataset):
    def __init__(self, df, tfms=None, size=256, tta=1, cfg: Config=None, test=False, scale=None, prefix='train'):
        self.df = df
        self.tfms = tfms
        self.size = size
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.cfg = cfg
        self.scale = scale or []
        self.tensor_tfms = Compose([
            ToTensor(),
            # Normalize(mean=[0.485, 0.456, 0.406,0.406], std=[0.229, 0.224, 0.225,0.225])
            Normalize(mean=[0.08323, 0.05766, 0.0554, 0.08365], std=[0.13553, 0.09504, 0.14648, 0.1332]),
        ])
        self.tta = tta
        self.test = test
        # self.prefix = prefix
        # self.seq_cvt = {x.split('_')[-1].split('.')[0]: x.split('/')[-1]
        #                 for x in glob.glob(str(self.path / '../../input/{}/*/*/*.jpg'.format(self.prefix)))}
        #

    def __len__(self):
        return self.df.shape[0] * self.tta

    def __getitem__(self, idx: int):
        item = self.df.iloc[idx % self.df.shape[0]]
        path = str(self.path / '../../input/train_images/{}'.format(
            item['image_id']
        ))
        # print(path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not self.cfg.transform.size == 512:
            # indeed size are 800 x 600
            # however, a error set default as 512
            img = cv2.resize(img, (self.cfg.transform.size, self.cfg.transform.size))
        if self.tfms:
            return (
                self.tensor_tfms(self.tfms(image=img)['image']),
                item.label
            )
        else:
            return (
                self.tensor_tfms(img),
                item.label
            )


class RANZERDataset(Dataset):
    def __init__(self, df, tfms=None, cfg=None, mode='train', file_dict=None):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = tfms
        self.cfg = cfg
        # REVERT THIS FOUR CHANNEL FOR BBBC
        self.tensor_tfms = Compose([
            # ToTensor(),
            # Normalize(mean=[0.485, 0.456, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225])
            Normalize(mean=[0.08323, 0.05766, 0.0554, 0.08365], std=[0.13553, 0.09504, 0.14648, 0.1332]),
        ])
        # # For HPA
        # self.tensor_tfms = Compose([
        #     # ToTensor(),
        #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.file_dict = file_dict
        self.cols = ['class{}'.format(i) for i in range(19)]
        self.cell_path = self.cfg.data.dir

    def __len__(self):
        return len(self.df)

    # def tensor_tfms(self, img):
    #     # convert PIL image to numpy array
    #     img_np = np.array(img) #last dimension is channel 
    #     output = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))
    #     mean = np.mean(img_np, axis=(0,1))
    #     std = np.std(img_np, axis=(0,1))
    #     transform_norm = transforms.Compose([
    #         transforms.ToTensor(),])
    #     # get normalized image
    #     output = transform_norm(img_np)
    #     mean = torch.mean(output)
    #     std = torch.std(output)
        
    #     cv2.imwrite("pic.png", np.transpose(np.array(output),(0,2,1)))
    #     return output # transform_norm(np.transpose(np.array(output),(1,0,2)))

    def __getitem__(self, index):
        if self.mode == 'train':
            row = self.df.loc[index]
            cnt = self.cfg.experiment.count
            cells = self.cfg.experiment.num_cells
            batch = torch.zeros((cells, self.cfg.experiment.n_channels, self.cfg.transform.size, self.cfg.transform.size))
            mask = np.zeros((cells))
            label = np.zeros((cells, 19)) 
            
            # color_dict = {1: 'blue', 2: 'green', 3: 'red_merged'} #, 4: 'yellow'}
            img = torch.zeros((cells, self.cfg.experiment.n_channels, self.cfg.transform.size, self.cfg.transform.size))
            for i in range(cells):
                path_channel = self.path / f'../../{self.cell_path}/{row["ID"]}_cell{i+1}.png'
                img_cell = imread(path_channel)
                # print(f'img_cell: {img_cell.shape}')
                img_cell = np.transpose(cv2.resize(img_cell, (self.cfg.transform.size, self.cfg.transform.size)),(2,0,1))
                
                if self.transform is not None:
                    res = self.transform(image=img_cell)
                    img_cell = res['image']
                    
                # print(f'img_cell:{img_cell.shape}')
                img_cell = self.tensor_tfms(torch.tensor(img_cell).double())
                # img_cell = torch.transpose(img_cell, 1, 0)
                img_cell = img_cell.unsqueeze(0)
                # print(f'img_cell: {img_cell.shape}') # torch.Size([1, 4, 256, 256])
                img[i, :, :, :] = img_cell
    
                batch = img # 10 x 4 x 256 x 256
                mask[i] = 1
                label[i] = row[self.cols].values.astype(np.float64)
                # print(f'batch size: {batch.shape}') # torch.Size([10, 4, 256, 256])
                # img = self.tensor_tfms(img)
            if self.cfg.experiment.smoothing == 0:
                # print(f'batch: {batch}, mask: {mask}, label: {label}')
                return batch, mask, label, row[self.cols].values.astype(np.float64)
            else:
                return batch, mask, 0.9*label + 0.1/19, 0.9 * row[self.cols].values.astype(np.float64) + 0.1/19

class ValidationDataset(Dataset):
    def __init__(self, df, tfms=None, cfg=None, mode='valid', file_dict=None):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = tfms
        self.cfg = cfg
        self.tensor_tfms = Compose([
            # ToTensor(),
            # Normalize(mean=[0.485, 0.456, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225])
            Normalize(mean=[0.08323, 0.05766, 0.0554, 0.08365], std=[0.13553, 0.09504, 0.14648, 0.1332]),
        ]) # 4 channels
        # self.tensor_tfms = Compose([
        #     # ToTensor(),
        #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ]) # 3 channels
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.file_dict = file_dict
        self.cols = ['class{}'.format(i) for i in range(19)]
        self.cell_path = self.cfg.data.dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.mode == 'valid':
            row = self.df.loc[index]
            cnt = self.cfg.experiment.num_cells
            cells = self.cfg.experiment.num_cells
            batch = torch.zeros((cnt, cnt*cells, self.cfg.experiment.n_channels, self.cfg.transform.size, self.cfg.transform.size))
            mask = np.zeros((cnt))
            label = np.zeros((cnt, 19)) 
            
            # color_dict = {1: 'blue', 2: 'green', 3: 'red_merged'} #, 4: 'yellow'}
            img = torch.zeros((cells, self.cfg.experiment.n_channels, self.cfg.transform.size, self.cfg.transform.size))
            for i in range(cells):
                path_channel = self.path / f'../../{self.cell_path}/{row["ID"]}_cell{i+1}.png'
                img_cell = imread(path_channel)
                img_cell = np.transpose(cv2.resize(img_cell, (self.cfg.transform.size, self.cfg.transform.size)),(2,0,1))
                if self.transform is not None:
                    res = self.transform(image=img_cell)
                    img_cell = res['image']

                img_cell = self.tensor_tfms(torch.tensor(img_cell).double())
                img[i, :, :, :] = img_cell
                batch = img # #10 x 4 x 256 x 256
                mask[i] = 1
                label[i] = row[self.cols].values.astype(np.float64)
            return batch, mask, label, row[self.cols].values.astype(np.float64), cnt


class TestDataset(Dataset):
    def __init__(self, df, tfms=None, tta=1, cfg=None, mode='test', file_dict=None):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = tfms
        self.cfg = cfg
        self.tta = tta
        self.tensor_tfms = Compose([
            # ToTensor(),
            # Normalize(mean=[0.485, 0.456, 0.406,0.406], std=[0.229, 0.224, 0.225,0.225])
            Normalize(mean=[0.08323, 0.05766, 0.0554, 0.08365], std=[0.13553, 0.09504, 0.14648, 0.1332]),
        ]) # 4 channels
        # self.tensor_tfms = Compose([
        #     # ToTensor(),
        #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ]) # 3 channels
        self.path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.file_dict = file_dict
        self.cols = ['class{}'.format(i) for i in range(19)]
        self.cell_path = self.cfg.data.dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.mode == 'test':
            # print(f'row: {self.df.loc[index]}')
            row = self.df.loc[index]
            cnt = self.cfg.experiment.num_cells
            cells = self.cfg.experiment.num_cells
            batch = torch.zeros((cnt, cnt*cells, self.cfg.experiment.n_channels, self.cfg.transform.size, self.cfg.transform.size))
            mask = np.zeros((cnt))
            label = np.zeros((cnt, 19)) 
            
            # color_dict = {1: 'blue', 2: 'green', 3: 'red_merged'} #, 4: 'yellow'}
            img = torch.zeros((cells, self.cfg.experiment.n_channels, self.cfg.transform.size, self.cfg.transform.size))
            for i in range(cells):
                path_channel = self.path / f'../../{self.cell_path}/{row["ID"]}_cell{i+1}.png'
                img_cell = imread(path_channel)
                img_cell = np.transpose(cv2.resize(img_cell, (self.cfg.transform.size, self.cfg.transform.size)),(2,0,1))
                if self.transform is not None:
                    res = self.transform(image=img_cell)
                    img_cell = res['image']

                img_cell = self.tensor_tfms(torch.tensor(img_cell).double())
                img[i, :, :, :] = img_cell
                batch = img # #10 x 4 x 256 x 256
                mask[i] = 1
                label[i] = row[self.cols].values.astype(np.float64)
            return batch, mask, label, row[self.cols].values.astype(np.float64), cnt
