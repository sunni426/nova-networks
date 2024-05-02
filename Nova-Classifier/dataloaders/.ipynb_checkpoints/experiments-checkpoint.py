# nova-networks/HPA-nova
from utils import Config
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from dataloaders.datasets import STRDataset, LandmarkDataset, RANZERDataset, ValidationDataset, TestDataset
# from dataloaders.transform import get_tfms
from dataloaders.transform_loader import get_tfms
from collections import Counter, defaultdict
import pickle
import os
import numpy as np
from dataloaders.sampler import RandomBatchSampler
import random
import albumentations as A
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop, RandomResizedCrop,
    RandomHorizontalFlip, RandomAffine, RandomVerticalFlip, RandomChoice, ColorJitter, RandomRotation)
import json
import uuid
import torch
from dataloaders.datasets import a_ordinary_collect_method


class RandomKTrainTestSplit:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        path = Path(os.path.dirname(os.path.realpath(__file__)))
        csv_file = cfg.experiment.file
        train = pd.read_csv(path / 'split' / csv_file)
        valid = pd.read_csv(path / 'split' / cfg.experiment.csv_valid)
        test = pd.read_csv(path / 'split' / cfg.experiment.csv_test)
        # # start
        # FOR RUNNING FOLD, commented out 2/11
        # self.train_meta, self.valid_meta = (train[train.fold != cfg.experiment.run_fold],
        #                                     valid[valid.fold != cfg.experiment.run_fold])

        self.train_meta, self.valid_meta, self.test_meta = (train[:], valid[:], test[:])
        print(f'meta:{train[:]}')

        # print(train.head())
        if cfg.basic.debug:
            print('[ W ] Debug Mode!, down sample')
            self.train_meta = self.train_meta.sample(frac=0.05)
            self.valid_meta = self.valid_meta.sample(frac=0.05)
            self.test_meta = self.test_meta.sample(frac=0.05)

    def get_dataloader(self, test_only=False, train_shuffle=True, infer=False, tta=-1, tta_tfms=None):
        if test_only:
            # raise NotImplementedError()
            # start
            if tta == -1:
                tta = 1
            # path = Path(os.path.dirname(os.path.realpath(__file__)))
            # test = pd.read_csv(path / '../../ {cfg.csv_valid}') # Change to new test dataset later
            # print(f'self.test_meta: {self.test_meta}')
            test_tfms = get_tfms(self.cfg.transform.test_name)
            test_ds = TestDataset(df=self.test_meta, tfms=test_tfms, tta=tta, cfg=self.cfg, mode='test')
            test_dl = DataLoader(dataset=test_ds, batch_size=1,
                              collate_fn=a_ordinary_collect_method, drop_last=True,
                              num_workers=self.cfg.transform.num_preprocessor, pin_memory=True)

            # print(f'self.valid_meta: {self.test_meta}, self.valid_meta: {self.test_meta}')

            
            # test_ds = ValidationDataset(df=self.test_meta, tfms=test_tfms, cfg=self.cfg, mode='valid')
            # test_dl = DataLoader(dataset=test_ds, batch_size=1,
            #                   collate_fn=a_ordinary_collect_method, drop_last=True,
            #                   num_workers=self.cfg.transform.num_preprocessor, pin_memory=True)

        
            # test['prefix'] = 'kaggle'
            # test_ds = SIIMDataset(test, tta_tfms, size=self.cfg.transform.size, tta=tta, test=True)
            # test_dl = DataLoader(dataset=test_ds, batch_size=self.cfg.eval.batch_size,
            #                       num_workers=self.cfg.transform.num_preprocessor, pin_memory=True)
            return test_dl
        # end
        print('[ √ ] Using transformation: {} & {}, image size: {}'.format(
            self.cfg.transform.name, self.cfg.transform.val_name, self.cfg.transform.size
        ))
        if self.cfg.transform.name == 'None':
            train_tfms = None
        else:
            train_tfms = get_tfms(self.cfg.transform.name)
        if tta_tfms:
            val_tfms = tta_tfms
        elif self.cfg.transform.val_name == 'None':
            val_tfms = None
        else:
            val_tfms = get_tfms(self.cfg.transform.val_name)
        # augmentation end
        # def __init__(self, df, tfms=None, cfg=None, mode='train'):
        # train_ds = STRDataset(self.train_meta, train_tfms, size=self.cfg.transform.size,
        #                       cfg=self.cfg, prefix=self.cfg.experiment.preprocess)
        train_ds = RANZERDataset(df=self.train_meta, tfms=train_tfms,
                                 cfg=self.cfg, mode='train')
        if self.cfg.experiment.weight and train_shuffle:
            train = self.train_meta.copy()
            method_dict = {
                'sqrt': np.sqrt,
                'log2': np.log2,
                'log1p': np.log1p,
                'log10': np.log10,
                'as_it_is': lambda w: w
            }
            if self.cfg.experiment.method in method_dict:
                print('[ √ ] Use weighted sampler, method: {}'.format(self.cfg.experiment.method))
                # HPA!!!
                # cw = (1 / method_dict[self.cfg.experiment.method](train.iloc[:, 1:19].sum(0))).values
                # print(train.head(2))
                # weight = (train.iloc[:, 1:19] * cw).max(1).values

                # BBBC!!!
                # cw = (1 / method_dict[self.cfg.experiment.method](train.iloc[:, 3:695].sum(0))).values
                print(train.head(2))
                weight = (train.iloc[:, 3:695] * 1).max(1).values
                

            
                # print(weight)
                # weight = 1 / method(train.groupby('label').transform('count').image_id.values)
            elif 'pow' in self.cfg.experiment.method:
                p = float(self.cfg.experiment.method.replace('pow_', ''))
                print('[ √ ] Use weighted sampler, method: Power of {}'.format(p))
                for x in ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']:
                    train['{}_p'.format(x)] = (1 / np.power(
                        train[[x, 'fold']].groupby(x).transform('count')['fold'].values, p)
                                               ) / len(train[x].value_counts())
                weight = train[['grapheme_root_p', 'vowel_diacritic_p', 'consonant_diacritic_p']].max(1).values
            else:
                raise Exception('Unknown weighting method!')
            rs = WeightedRandomSampler(weights=weight, num_samples=len(weight))
            train_dl = DataLoader(train_ds, sampler=rs, batch_size=self.cfg.train.batch_size,
                                  num_workers=self.cfg.transform.num_preprocessor, pin_memory=True)
        elif self.cfg.experiment.batch_sampler:
            print('[ i ] Batch Sampler!')
            bs = RandomBatchSampler(train_ds.df, self.cfg.train.batch_size, cfg=self.cfg)
            train_dl = DataLoader(dataset=train_ds, batch_sampler=bs,
                                  num_workers=self.cfg.transform.num_preprocessor)
        else:
            train_dl = DataLoader(dataset=train_ds, batch_size=self.cfg.train.batch_size,
                                  num_workers=self.cfg.transform.num_preprocessor,
                                  shuffle=train_shuffle, drop_last=True, pin_memory=True)
            # print(f'dl:{train_dl}')
        if tta == -1:
            tta = 1
        # valid_ds = STRDataset(self.valid_meta, val_tfms, size=self.cfg.transform.size, tta=tta,
        #                       cfg=self.cfg, prefix=self.cfg.experiment.preprocess)
        # def __init__(self, df, tfms=None, cfg=None, mode='train'):


        # valid_ds = RANZERDataset(df=self.valid_meta, tfms=val_tfms, cfg=self.cfg,
                                 # mode='valid') # consider rewriting to manually load
        valid_ds = ValidationDataset(df=self.valid_meta, tfms=val_tfms, cfg=self.cfg, mode='valid')
        valid_dl = DataLoader(dataset=valid_ds, batch_size=self.cfg.eval.batch_size,
                              collate_fn=a_ordinary_collect_method, drop_last=True,
                              num_workers=self.cfg.transform.num_preprocessor, pin_memory=True)
        
        return train_dl, valid_dl, None