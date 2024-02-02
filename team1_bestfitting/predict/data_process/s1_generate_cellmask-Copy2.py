import sys
sys.path.insert(0, '..')
import glob
import imageio
import argparse
from tqdm import tqdm
import pandas as pd
from PIL import Image
from mlcrate import SuperPool
from functools import partial
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

from config.config import *
import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell, label_nuclei
from utilities.result_util import encode_binary_mask
from utilities.data_util import generate_cell_indices
import pytorch_zoo
import torch
torch.nn.Module.dump_patches = True
import matplotlib.pyplot as plt
from IPython.display import Image, display

def encode_mask(param):
  cell_mask_fname = param

  # ADDED: Load the image
  image = cv2.imread(param)

  # Check if the image is successfully loaded
  if image is not None:
    # Print the pixel values
    print("Pixel values:")
    print(image)
  else:
    print(f"Error: Unable to load the image from '{image_path}'. Check the file path.")
    
  # display(Image(param))
  img_name = os.path.basename(cell_mask_fname).replace('_cellmask.png', '')
  
  cellmask = cv2.imread(cell_mask_fname, flags=cv2.IMREAD_GRAYSCALE)
  cell_indices = generate_cell_indices(cellmask)
  print(cellmask)
  print(f'img_name is {img_name} and cellmask is {cellmask}')

  df = []
  for cell_index in cell_indices:
    cellmask_encode = encode_binary_mask(cellmask == cell_index)
    df.append([img_name, cellmask.shape[0], cellmask.shape[1], cell_index, cellmask_encode])
  df = pd.DataFrame(df, columns=[ID, HEIGHT, WIDTH, 'maskid', 'cellmask'])
  return df

def generate_labels(row):
  labels = np.array(str(row[LABEL]).split('|'), dtype='int32')
  for label in labels:
    row[ALIASES[label]] = 1
  return row

def generate_image_shape(image_id, suffix='png', dataset='train'):
  if suffix in ['png', 'jpg']:
    image = cv2.imread(f'{DIR_CFGS.DATA_DIR}/images/{dataset}/{image_id}_red.{suffix}', flags=cv2.IMREAD_GRAYSCALE)
  else:
    raise ValueError(suffix)
  for color in COLORS:
    if not ope(f'{DIR_CFGS.DATA_DIR}/images/{dataset}/{image_id}_{color}.{suffix}'):
      image = None
      break

  if image is None:
    shape = [None, None]
  else:
    shape = list(image.shape[:2])
  return shape

def load_train_data():
  data_df = pd.read_csv(f'{DIR_CFGS.DATA_DIR}/raw/train.csv')
  print(data_df.head(), sum(data_df.Label.isna()))
  return data_df

def load_test_data():
  data_df = pd.read_csv(f'{DIR_CFGS.DATA_DIR}/raw/test.csv')
  print(sum(data_df.Label.isna()))
  print(data_df[data_df.Label.isna()])
  data_df = data_df[~data_df.Label.isna()]
  return data_df

def generate_inputs(output_dir, pool, dataset='train'):
  data_df = eval(f'load_{dataset}_data')()
  for col in ALIASES: data_df[col] = 0
  data_df = data_df.apply(generate_labels, axis=1)

  generate_image_shape_func = partial(generate_image_shape, suffix='png', dataset=dataset)
  shapes = pool.map(generate_image_shape_func, data_df[ID].values)
  shapes = np.array(shapes)
  data_df.insert(2, WIDTH, shapes[:, 1])
  data_df.insert(3, HEIGHT, shapes[:, 0])
  data_df = data_df[data_df[WIDTH].notnull() & data_df[HEIGHT].notnull()]
  data_df = data_df.reset_index(drop=True)

  print(data_df.shape)
  print(data_df.head())
  print(data_df[ALIASES].sum(axis=0).sort_values())

  os.makedirs(output_dir, exist_ok=True)
  data_df.to_csv(f'{output_dir}/{dataset}.csv', index=False, encoding='utf-8')

def segment_image(segmentator, param):
    images, save_dir = param
    print("IMAGE IN SEGMENT:",images)
    mask_name = os.path.basename(images[0][0]).replace('red', 'cellmask')
    if ope(f'{save_dir}/{mask_name}'): #deb thinks it ope(f.....), ope=os.path.existss
        return
    try:        
        # For nuclei
        images=np.array(images) #rescale() got an unexpected keyword argument 'multichannel'
        # images has 3 channels: red, yellow, blue
        images = [item[0] for item in images]
        nuc_segmentations = segmentator.pred_nuclei(images)
        print("try works")
        # For full cells
        cell_segmentations = segmentator.pred_cells(images)
    except Exception as e: 
        print(e) #'list' object has no attribute 'shape' 
        img = imageio.imread(images[0]) # channel 9
        print(mask_name, img.shape)
        print('except works')
    else:
        # post-processing
        print('else works')
        for i, _ in enumerate(cell_segmentations):
            nuclei_mask, cell_mask = label_cell(nuc_segmentations[i], cell_segmentations[i])
            # print(f'images[0][i]: {images[0][i]}')
            # print(f'nuc_segmentations[i]: {nuc_segmentations[i]}')
            # print(f'images[0]: {images[0]}')
            # mask_name = os.path.basename(images[0][i]).replace('red', 'cellmask')
            mask_name = os.path.basename(images[0]).replace('red', 'cellmask')
            nuclei_mask_name = mask_name.replace('cellmask', 'nucleimask')
            print(f'\n\nmask_name: {mask_name}\n\n')
            print(f'\n\n{save_dir}/{mask_name}\n\n')
            imageio.imwrite(f'{save_dir}/{mask_name}', cell_mask)
            imageio.imwrite(f'{save_dir}/{nuclei_mask_name}', nuclei_mask)

            
            # Check if the image is successfully loaded
            if cell_mask is not None:
                # Print the pixel values
                print("Pixel values for cell_mask:")
                print(cell_mask)
            else:
                print(f"Error: Unable to load the image from '{image_path}'. Check the file path.")
    


              
              
        
def generate_cellmask_csv(dataset, pool):
  mask_dir = f'{DIR_CFGS.DATA_DIR}/mask/{dataset}'
  cell_mask_fnames = glob.glob(f'{mask_dir}/*_cellmask.png')
  print("df",cell_mask_fnames) #char


  if dataset == 'test':
    #df = pd.read_csv(f'{DIR_CFGS.DATA_DIR}/raw/sample_submission.csv')
    df = pd.read_csv(f'{DIR_CFGS.DATA_DIR}/raw/test.csv')
  else:
    df = pd.read_csv(f'{DIR_CFGS.DATA_DIR}/inputs/{dataset}.csv')
  img_ids = df[ID].unique()
  params = []
  for cell_mask_fname in cell_mask_fnames:
    img_id = cell_mask_fname.split('/')[-1].replace('_cellmask.png', '')
    print('hiiiii',img_id)
    if img_id in img_ids:
      params.append(cell_mask_fname)

  # print(f'encode_mask: {encode_mask}')

  mask_df = encode_mask(params[0])
  # print(f'encode_mask: {encode_mask}')
    
  # mask_df = pool.map(encode_mask, params)
  print('HERE:',mask_df) #added by deb
    
  mask_df = pd.concat(mask_df).reset_index(drop=True, inplace=True)
  # print(f'img num:{mask_df[ID].nunique()}, cell num: {len(mask_df)}')
  print('mask_df:',mask_df)
  # print(f'img num:{mask_df}, cell num: {len(mask_df)}')
  print(mask_df.head())
  mask_df.to_csv(f'{DIR_CFGS.DATA_DIR}/mask/{dataset}.csv', index=False, encoding='utf-8')

def generate_cellmask(args, split_df=None, mask_output_dir=None):
  segmentator = cellsegmentator.CellSegmentator(
    NUC_MODEL,
    CELL_MODEL,
    scale_factor=0.25,
    device=f'cuda:{args.gpus}',
    padding=True,
    multi_channel_model=True,
  )
  dataset = args.dataset
  print(f'segment: {dataset}')
  if dataset in ['train', 'test']:
    image_dir = f'{DIR_CFGS.DATA_DIR}/images/{dataset}'
  else:
    image_dir = f'{DIR_CFGS.DATA_DIR}/images/{dataset}_png'

  if mask_output_dir is None:
    save_dir = f'{DIR_CFGS.DATA_DIR}/mask/{dataset}'
  else:
    save_dir = mask_output_dir
  print(f'save_dir: {save_dir}')
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  if split_df is None:
    mt = glob.glob(f'{image_dir}/*_red.png')
    mt.sort()
  else:
    mt = np.asarray([opj(image_dir, f'{_id}_red.png') for _id in split_df[ID].values])
  er = [f.replace('red', 'yellow') for f in mt]
  nu = [f.replace('red', 'blue') for f in mt]
  images = [mt, er, nu]

  for i in tqdm(range(len(images[0])), total=len(images[0])):
    images_temp = [[images[0][i]], [images[1][i]], [images[2][i]]]
    param = (images_temp, save_dir)
    segment_image(segmentator, param)


def main(args, split_df=None, mask_output_dir=None):
  pool = SuperPool(n_cpu=24)
  output_dir = f'{DIR_CFGS.DATA_DIR}/inputs'
  generate_inputs(output_dir, pool, dataset=args.dataset)
  generate_cellmask(args, split_df, mask_output_dir)
  generate_cellmask_csv(dataset=args.dataset, pool=pool)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='segment cell')
  parser.add_argument('--gpus', default=None, type=str, help='use gpu')
  parser.add_argument('--dataset', default='train', type=str, help='dataset')
  args = parser.parse_args()
  main(args, split_df=None, mask_output_dir=None)

