import sys
sys.path.insert(0, '..')

import cv2
from mlcrate import SuperPool
import argparse

from config.config import *

def resize_images(src_dir, dst_dir, image_size, suffix, pool, fname_list=None):
  os.makedirs(dst_dir, exist_ok=True)
  if fname_list is None:
    fname_list = sorted(os.listdir(src_dir))

  def _resize_images(fname):
    src_fpath = f'{src_dir}/{fname}'
    out_fname = f'{fname.split(".")[0]}.{suffix}'
    dst_fpath = f'{dst_dir}/{out_fname}'
    if ope(dst_fpath):
      return
    try:
      if fname.count('yellow') | fname.count('red') | fname.count('green') | fname.count('blue'):
        image = cv2.imread(src_fpath, flags=cv2.IMREAD_GRAYSCALE)
      else:
        image = cv2.imread(src_fpath, flags=cv2.IMREAD_UNCHANGED)
      image = cv2.resize(image, (image_size[1], image_size[0]))
      cv2.imwrite(dst_fpath, image)
    except:
      print(f'error image: {src_fpath}')

  pool.map(_resize_images, fname_list, description='resize images')

def main():
  assert args.suffix in ['jpg', 'png']
  args.image_size = args.image_size.split(',')
  if len(args.image_size) == 1:
    args.image_size = [int(args.image_size[0]), int(args.image_size[0])]
  elif len(args.image_size) == 2:
    args.image_size = [int(args.image_size[0]), int(args.image_size[1])]
  else:
    raise ValueError(','.join(args.image_size))
  pool = SuperPool(n_cpu=24)

  if args.dataset in ['train', 'test']:
    src_dir = f'{DIR_CFGS.DATA_DIR}/images/{args.dataset}'
  else:
    src_dir = f'{DIR_CFGS.DATA_DIR}/images/{args.dataset}_png_i512x512'

  dst_dir = f'{DIR_CFGS.DATA_DIR}/images/{args.dataset}_{args.suffix}_i{args.image_size[0]}x{args.image_size[1]}'
  print(src_dir, '---to---', dst_dir)
  resize_images(src_dir, dst_dir, args.image_size, args.suffix, pool)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='PyTorch Classification')
  parser.add_argument('--image_size', type=str, default='512')
  parser.add_argument('--dataset', type=str, default='train')
  parser.add_argument('--suffix', type=str, default='jpg')
  args = parser.parse_args()
  main()
