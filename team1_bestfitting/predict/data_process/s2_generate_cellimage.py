import sys
sys.path.insert(0, '..')
import argparse
import cv2
import pandas as pd
from mlcrate import SuperPool
import imageio

from config.config import *
from utilities.data_util import load_rgby, load_mask, generate_cell_indices
from utilities.augment_util import cell_crop_augment

def crop_image(param):
  image_dir, cellmask_dir, save_dir, image_id, image_size, cell_type = param
  image_raw = load_rgby(image_dir, image_id, suffix='png')
  if (image_size > 0) & (image_raw.shape[0] != image_size):
    image_raw = cv2.resize(image_raw, (image_size, image_size))
  #mask_raw = load_mask(cellmask_dir, image_id) #In original code, wrong :(
  print('image_id',image_id)
  mask_raw=imageio.imread(f'{cellmask_dir}/{image_id}_cellmask.png')
  print('mask_raw: ',mask_raw)  
  if (image_size > 0) & (mask_raw.shape[0] != image_size): #both TRUE, image_size=128
    #mask_raw = cv2.resize(mask_raw, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    mask_raw = cv2.resize(mask_raw, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4) #INTER_LANCZOS4 have better quality 
  cell_indices = generate_cell_indices(mask_raw) #empty
  #print(cell_indices) 
    
  df = []
  
  for maskid in cell_indices:
    save_fname = f'{save_dir}/{image_id}_{maskid}.png'
    mask_area = (mask_raw == maskid).sum()
    if ope(save_fname):
      df.append([image_id, maskid, mask_area, np.sqrt(mask_area), np.sqrt(mask_area)])
      continue
    image = image_raw.copy()
    mask = mask_raw.copy()
    if cell_type == 2:
      image[mask == maskid] = 0
    else:
      image[mask != maskid] = 0
    if cell_type == 1:
      image, _ = cell_crop_augment(image, (mask == maskid).astype('uint8'))
    cv2.imwrite(save_fname, image)
    df.append([image_id, maskid, mask_area, image.shape[1], image.shape[0]])
  return pd.DataFrame(df, columns=[ID, 'maskid', 'mask_area', WIDTH, HEIGHT])

def main(args):
  if args.image_size != -1:
    save_dir = f'{DIR_CFGS.DATA_DIR}/images/{args.dataset}_cell_v{args.cell_type}_png_i{args.image_size}x{args.image_size}'
  else:
    save_dir = f'{DIR_CFGS.DATA_DIR}/images/{args.dataset}_cell_v{args.cell_type}_png'

  print(f'image save_dir: {save_dir}')
  os.makedirs(save_dir, exist_ok=True)
  if args.dataset in ['train', 'test']:
    image_dir = f'{DIR_CFGS.DATA_DIR}/images/{args.dataset}'
  else:
    image_dir = f'{DIR_CFGS.DATA_DIR}/images/{args.dataset}_png'
  cellmask_dir = f'{DIR_CFGS.DATA_DIR}/mask/{args.dataset}'
  print(f'image_dir: {image_dir}')
  print(f'cellmask_dir: {cellmask_dir}')

  if args.split_df is not None:
    df = args.split_df.copy()
  elif args.dataset == 'test':
    #df = pd.read_csv(f'{DIR_CFGS.DATA_DIR}/raw/sample_submission.csv')
    df = pd.read_csv(f'{DIR_CFGS.DATA_DIR}/raw/test.csv')
  else:
    df = pd.read_csv(f'{DIR_CFGS.DATA_DIR}/inputs/{args.dataset}.csv')

  params = []
  for image_id in df[ID].values:
    params.append((image_dir, cellmask_dir, save_dir, image_id, args.image_size, args.cell_type))

  if args.pool is None:
    pool = SuperPool(n_cpu=args.n_cpu)
  else:
    pool = args.pool
  mask_df = pool.map(crop_image, params) #Empty DataFrame, params not empty  
  mask_df = pd.concat(mask_df).reset_index(drop=True) 
  print(mask_df['mask_area'].describe())
  print(mask_df[WIDTH].describe())
  print(mask_df[HEIGHT].describe())
  mask_df.to_csv(f'{DIR_CFGS.DATA_DIR}/images/{args.dataset}_maskdf.csv')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='segment cell')
  parser.add_argument('--dataset', default='publichpa', type=str, help='dataset')
  parser.add_argument('--n_cpu', default=2, type=int, help='n_cpu')
  parser.add_argument('--cell_type', default=1, type=int)
  parser.add_argument('--image_size', default=-1, type=int)
  parser.add_argument('--pool', default=None, type=object)
  parser.add_argument('--split_df', default=None, type=object)
  args = parser.parse_args()
  main(args)
