import sys
sys.path.insert(0, '..')

import pandas as pd
import torch
from torch.utils.data import Dataset

from config.config import *
from utilities.args_util import set_default_attr
from utilities.dataset_util import process_dataset
from utilities.augment_util import *

class CellClsDataset(Dataset):
  def __init__(self, args, transform, dataset='train', is_training=True):
    self.args = set_default_attr(args)
    self.transform = transform
    self.dataset = dataset
    self.is_training = is_training

    self.init_dataset()

  def init_dataset(self):
    cell_version = '' if self.args.cell_version is None else self.args.cell_version
    if self.dataset in ['train', 'valid'] and self.is_training:
      self.split_fpath = f'{DIR_CFGS.DATA_DIR}/split/{self.args.split_type}_folds{self.args.folds_num}/random_{self.dataset}_cv{self.args.fold}.csv.gz'
    elif self.dataset in ['test'] and self.is_training:
      self.split_fpath = f'{DIR_CFGS.DATA_DIR}/inputs/cellv4b_test.csv'
    elif self.dataset in ['test']:
      self.split_fpath = f'{DIR_CFGS.DATA_DIR}/mask/{cell_version}/{self.dataset}.csv'
    else:
      self.split_fpath = f'{DIR_CFGS.DATA_DIR}/mask/{self.dataset}.csv'

    self.image_dir_list = [
      f'{DIR_CFGS.DATA_DIR}/images/train_cell_v{self.args.cell_type}_{self.args.suffix}_i{self.args.image_size[0]}x{self.args.image_size[1]}',
      f'{DIR_CFGS.DATA_DIR}/images/test_cell_v{self.args.cell_type}_{self.args.suffix}_i{self.args.image_size[0]}x{self.args.image_size[1]}',
      f'{DIR_CFGS.DATA_DIR}/images/publichpa_cell_v{self.args.cell_type}_{self.args.suffix}_i{self.args.image_size[0]}x{self.args.image_size[1]}',
      f'{DIR_CFGS.DATA_DIR}/images/HPAv19_cell_v{self.args.cell_type}_{self.args.suffix}_i{self.args.image_size[0]}x{self.args.image_size[1]}',
      f'{DIR_CFGS.DATA_DIR}/images/HPAv18_cell_v{self.args.cell_type}_{self.args.suffix}_i{self.args.image_size[0]}x{self.args.image_size[1]}',
      f'{DIR_CFGS.DATA_DIR}/images/train2019_cell_v{self.args.cell_type}_{self.args.suffix}_i{self.args.image_size[0]}x{self.args.image_size[1]}',
    ]

    # load data
    self.split_df = pd.read_csv(self.split_fpath)
    if self.args.whole_train and self.dataset == 'train':
      valid_split_fpath = self.split_fpath.replace('train', 'valid')
      valid_df = pd.read_csv(valid_split_fpath)
      self.split_df = pd.concat((self.split_df, valid_df), ignore_index=True)
    if self.args.split_df is not None:
      self.split_df = self.split_df[self.split_df[ID].isin(self.args.split_df[ID].values)]
    if 'cellmask' not in self.split_df: self.split_df['cellmask'] = None

    self.base_image_ids = self.split_df[ID].values
    self.base_maskids = self.split_df['maskid'].values
    self.base_cellmasks = self.split_df['cellmask'].values
    if self.is_training:
      if self.args.label is None:
        self.base_labels = self.split_df[ALIASES].values
      else:
        self.base_labels = self.split_df[[ALIASES[self.args.label]]].values
        if self.args.can_print:
          print(f'pos num: {self.base_labels.sum()}')

    self.resample(epoch=-1)

  def resample(self, epoch=0):
    self.image_ids = self.base_image_ids
    self.maskids = self.base_maskids
    self.cellmasks = self.base_cellmasks
    if self.is_training:
      self.labels = self.base_labels

  def set_part(self, part_start=0., part_end=1.):
    start_idx = int(np.round(len(self) * part_start))
    end_idx = int(np.round(len(self) * part_end))

    self.base_image_ids = self.base_image_ids[start_idx:end_idx]
    self.base_maskids = self.base_maskids[start_idx:end_idx]
    self.base_cellmasks = self.base_cellmasks[start_idx:end_idx]
    if self.is_training:
      self.base_labels = self.base_labels[start_idx:end_idx]
    self.resample(epoch=-1)

  def __len__(self):
    num = len(self.image_ids)
    if self.args.debug:
      num = min(num, 160)
    return num

  def load_image(self, image_dir_list, image_id):
    image = None
    for image_dir in image_dir_list:
      image = cv2.imread(f'{image_dir}/{image_id}.{self.args.suffix}', flags=cv2.IMREAD_UNCHANGED)
      #print('Readimage from:', f'{image_dir}/{image_id}.{self.args.suffix}')
      if image is not None:
        break
    return image

  def load_image_mask_labels(self, index):
    image_id = self.image_ids[index]
    maskid = self.maskids[index]

    image = self.load_image(self.image_dir_list, f'{image_id}_{maskid}')
    assert image is not None, f'{image_id}_{maskid}'
    image = cv2.resize(image, (self.args.image_size[1], self.args.image_size[0]))

    if self.args.in_channels == 2:
      image = image[:, :, 1:3]
    else:
      image = image[:, :, 0:self.args.in_channels]

    assert image.shape == (self.args.image_size[0], self.args.image_size[1], self.args.in_channels), image.shape

    data = {
      ID: image_id,
      'index': index,
      'image': image,
      'mask': np.zeros((image.shape[0], image.shape[1])),
      'maskid': maskid,
      'cellmask': self.cellmasks[index],
    }
    if self.is_training:
      data['labels'] = self.labels[index].copy()
    return data

  def __getitem__(self, index):
    data = self.load_image_mask_labels(index)
    if self.transform:
      data = self.transform(data, self.args)
    data = process_dataset(data, self.args)
    return data

def cls_collate_fn(batch):
  data = {k: [b[k] for b in batch] for k in batch[0].keys()}
  data['image'] = torch.stack(data['image'], dim=0)
  data['mask'] = torch.stack(data['mask'], dim=0)
  if 'labels' in data:
    data['labels'] = torch.stack(data['labels'], dim=0)
  return data
