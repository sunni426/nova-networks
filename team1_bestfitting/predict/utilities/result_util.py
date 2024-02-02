import sys
sys.path.insert(0, '..')
import base64
import numpy as np
import pandas as pd
from pycocotools import _mask as coco_mask
import typing as t
import zlib
import cv2
from tqdm import tqdm

from config.config import *

def encode_binary_mask(mask: np.ndarray) -> t.Text:
  """Converts a binary mask into OID challenge encoding ascii text."""

  # check input mask --
  if mask.dtype != bool:
    raise ValueError(
        "encode_binary_mask expects a binary mask, received dtype == %s" %
        mask.dtype)

  mask = np.squeeze(mask)
  if len(mask.shape) != 2:
    raise ValueError(
        "encode_binary_mask expects a 2d mask, received shape == %s" %
        mask.shape)

  # convert input mask to expected COCO API input --
  mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
  mask_to_encode = mask_to_encode.astype(np.uint8)
  mask_to_encode = np.asfortranarray(mask_to_encode)

  # RLE encode mask --
  encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

  # compress and base64 encoding --
  binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
  base64_str = base64.b64encode(binary_str)
  return base64_str.decode()

def decode_binary_mask(base64_str: t.Text, width: int=512, height: int=512) -> np.ndarray:
  # base64 dencoding and decompress --
  base64_str = base64_str.encode()
  binary_str = base64.b64decode(base64_str)
  encoded_mask = zlib.decompress(binary_str)

  # mask decode RLE --
  mask_to_encode = coco_mask.decode([{'size': [height, width], 'counts': encoded_mask}])

  # convert COCO API input to input mask --
  mask_to_encode = mask_to_encode.astype(bool)
  mask = np.squeeze(mask_to_encode)

  return mask

def split_result_cell(result_df):
  result_df_list = []
  for image_idx in tqdm(range(len(result_df)), total=len(result_df), desc='split result'):
    image_id = result_df[ID].iloc[image_idx]
    width = result_df[WIDTH].iloc[image_idx]
    height = result_df[HEIGHT].iloc[image_idx]
    targets = np.array(result_df[TARGET].iloc[image_idx].split(' ')).reshape(-1, 3)

    target_dict = {}
    for label, conf, encoded_mask in targets:
      probs = target_dict.get(encoded_mask, np.zeros(NUM_CLASSES))
      probs[int(label)] = float(conf)
      target_dict[encoded_mask] = probs

    encoded_masks = sorted(list(target_dict.keys()))
    probs = np.array([target_dict[encoded_mask] for encoded_mask in encoded_masks])
    image_df = pd.DataFrame(data=probs, columns=ALIASES)
    image_df.insert(0, ID, image_id)
    image_df.insert(1, WIDTH, width)
    image_df.insert(2, HEIGHT, height)
    image_df.insert(3, 'mask', encoded_masks)

    result_df_list.append(image_df)
  result_df = pd.concat(result_df_list)
  return result_df

def merge_result_cell(result_df, labels=None):
  if labels is None:
    assert np.all(result_df[ALIASES].notnull())
  else:
    assert np.all(result_df[ALIASES[labels]].notnull())
  result_df = result_df.set_index(ID, drop=False)

  if labels is None:
    labels = range(NUM_CLASSES)
  elif isinstance(labels, int):
    labels = [labels]
  else:
    assert isinstance(labels, (list, tuple))
  result_list = []
  for image_id in tqdm(result_df[ID].unique(), total=result_df[ID].nunique(), desc='merge result'):
    image_df = result_df.loc[[image_id]].reset_index(drop=True)
    width = image_df[WIDTH].iloc[0]
    height = image_df[HEIGHT].iloc[0]

    target_list = []
    for mask_idx in range(len(image_df)):
      encoded_mask = image_df['mask'].iloc[mask_idx]
      for label in labels:
        prob = image_df[ALIASES[label]].iloc[mask_idx]
        target = f'{label} {prob} {encoded_mask}'
        target_list.append(target)
    target = ' '.join(target_list)
    result_list.append([image_id, width, height, target])
  result_df = pd.DataFrame(data=result_list, columns=[ID, WIDTH, HEIGHT, TARGET])

  return result_df

def set_text(image, label, is_gt=None):
  h, w = int(image.shape[0] * 0.15), image.shape[1]
  font = cv2.FONT_HERSHEY_COMPLEX

  if type(label) == int:
    label_names = [NAMES[label]]
    label = [label]
  else:
    label_names = [NAMES[l] for l in label]
  if is_gt is None:
    is_gt = [True] * len(label)
  for i, (l, name, gt) in enumerate(zip(label, label_names, is_gt)):
    if gt:
      cv2.putText(image, f'({l:02d}){name}', (int(h * 0.05), (i + 1) * int(h * 0.15)), font, 0.6, (255, 255, 255), 2, bottomLeftOrigin=False)
    else:
      cv2.putText(image, f'({l:02d}){name}', (int(h * 0.05), (i + 1) * int(h * 0.15)), font, 0.6, (0, 255, 255), 2, bottomLeftOrigin=False)
  return image

if __name__ == '__main__':
  height = 512
  width = 768
  mask = np.random.random((height, width))
  mask = mask > 0.5

  encode_mask = encode_binary_mask(mask)
  mask2 = decode_binary_mask(encode_mask, width=width, height=height)
  print(np.array_equal(mask, mask2))
