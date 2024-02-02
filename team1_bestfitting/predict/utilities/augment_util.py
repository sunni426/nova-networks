import sys
sys.path.insert(0, '..')
import cv2

from config.config import *
from skimage import measure

def cls_train_multi_augment1(data, args):
  augment_func_list = [
    cls_augment_default,              # default
    cls_augment_flipud,               # up-down
    cls_augment_fliplr,               # left-right
    cls_augment_transpose,            # transpose
    cls_augment_flipud_lr,            # up-down left-right
    cls_augment_flipud_transpose,     # up-down transpose
    cls_augment_fliplr_transpose,     # left-right transpose
    cls_augment_flipud_lr_transpose,  # up-down left-right transpose
  ]
  c = np.random.choice(len(augment_func_list))
  data = augment_func_list[c](data, args)
  return data

def cls_train_multi_augment2(data, args):
  data = cls_train_multi_augment1(data, args)
  if np.random.random() < 0.5:
    data = cls_augment_scale_rotate(data, args)
  return data

def cls_train_multi_augment6(data, args):
  if 'cell' in data:
    flags = False
    image = data['image'].copy()
    mask = data['mask'].copy()
    for i in range(len(data['cell']['image'])):
      image, mask, flag = random_select_position2(data['cell']['image'][i], data['cell']['mask'][i],
                                                  image, mask, data['cell']['maskid'][i])
      flags += flag
    if flags > 0:
      if args.label is not None:
        data['labels'] = np.maximum(data['labels'], data['cell']['label'][args.label])
      else:
        data['labels'] = np.maximum(data['labels'], data['cell']['label'])

    del data['cell']
    data['image'] = image
    data['mask'] = mask

  data = cls_train_multi_augment2(data, args)
  return data

def cls_augment_default(data, args):
  return data

def cls_augment_flipud(data, args):
  image_dtype = data['image'].dtype
  mask_dtype = data['mask'].dtype
  image = np.concatenate([data['image'], data['mask'][..., None]], axis=-1)
  image = np.flipud(image)

  data['image'] = image[..., :-1].astype(image_dtype)
  data['mask'] = image[..., -1].astype(mask_dtype)
  return data

def cls_augment_fliplr(data, args):
  image_dtype = data['image'].dtype
  mask_dtype = data['mask'].dtype
  image = np.concatenate([data['image'], data['mask'][..., None]], axis=-1)
  image = np.fliplr(image)

  data['image'] = image[..., :-1].astype(image_dtype)
  data['mask'] = image[..., -1].astype(mask_dtype)
  return data

def cls_augment_transpose(data, args):
  image_dtype = data['image'].dtype
  mask_dtype = data['mask'].dtype
  image = np.concatenate([data['image'], data['mask'][..., None]], axis=-1)
  image = np.transpose(image, (1, 0, 2))

  data['image'] = image[..., :-1].astype(image_dtype)
  data['mask'] = image[..., -1].astype(mask_dtype)
  return data

def cls_augment_flipud_lr(data, args):
  data = cls_augment_flipud(data, args)
  data = cls_augment_fliplr(data, args)
  return data

def cls_augment_flipud_transpose(data, args):
  data = cls_augment_flipud(data, args)
  data = cls_augment_transpose(data, args)
  return data

def cls_augment_fliplr_transpose(data, args):
  data = cls_augment_fliplr(data, args)
  data = cls_augment_transpose(data, args)
  return data

def cls_augment_flipud_lr_transpose(data, args):
  data = cls_augment_flipud(data, args)
  data = cls_augment_fliplr(data, args)
  data = cls_augment_transpose(data, args)
  return data

def cls_augment_scale_rotate(data, args, scales=(3/4., 4/3.), angles=(0., 360.)):
  assert scales[1] > scales[0]
  assert angles[1] > angles[0]

  image_dtype = data['image'].dtype
  mask_dtype = data['mask'].dtype
  image = np.concatenate([data['image'], data['mask'][..., None]], axis=-1)
  height, width, channel = image.shape

  # do scale
  scale = np.random.random() * (scales[1] - scales[0]) + scales[0]
  scale_width = int(np.round(width * scale))
  scale_height = int(np.round(height * scale))
  image, mask = image[..., :-1], image[..., -1]
  image = cv2.resize(image, (scale_width, scale_height), interpolation=cv2.INTER_LINEAR)
  mask = cv2.resize(mask, (scale_width, scale_height), interpolation=cv2.INTER_NEAREST)
  image = np.concatenate([image, mask[..., None]], axis=-1)
  scale_height, scale_width, channel = image.shape

  # do rotate
  angle = np.random.random() * (angles[1] - angles[0]) + angles[0]
  positions = np.array([
    [-scale_width / 2., -scale_height / 2.],
    [ scale_width / 2., -scale_height / 2.],
    [ scale_width / 2.,  scale_height / 2.],
    [-scale_width / 2.,  scale_height / 2.]
  ])
  matrix = np.array([
    [np.cos(angle * np.pi / 180), -np.sin(angle * np.pi / 180)],
    [np.sin(angle * np.pi / 180),  np.cos(angle * np.pi / 180)],
  ])
  new_positions = np.dot(matrix, positions.T).T
  rotate_width = int(np.round(new_positions[:, 0].max() - new_positions[:, 0].min()))
  rotate_height = int(np.round(new_positions[:, 1].max() - new_positions[:, 1].min()))
  M = cv2.getRotationMatrix2D((rotate_width / 2, rotate_height / 2), angle, 1)
  image2 = np.zeros((rotate_height, rotate_width, channel), dtype=image.dtype)
  image2[
    int(np.round((rotate_height - scale_height) / 2)):int(np.round((rotate_height - scale_height) / 2)) + scale_height,
    int(np.round((rotate_width - scale_width) / 2)):int(np.round((rotate_width - scale_width) / 2)) + scale_width,
  ] = image
  image, mask = image2[..., :-1], image2[..., -1]
  image = cv2.warpAffine(image, M, (rotate_width, rotate_height), borderMode=cv2.INTER_LINEAR)
  mask = cv2.warpAffine(mask, M, (rotate_width, rotate_height), borderMode=cv2.INTER_NEAREST)
  image = np.concatenate([image, mask[..., None]], axis=-1)

  # do crop
  new_height, new_width, channel = image.shape
  if new_height > height:
    starty = np.random.choice(new_height - height + 1)
    endy = starty + height
    image = image[starty:endy]
  elif height > new_height:
    starty = np.random.choice(height - new_height + 1)
    endy = starty + new_height
    image2 = np.zeros((height, image.shape[1], channel), dtype=image.dtype)
    image2[starty:endy] = image
    image = image2

  if new_width > width:
    startx = np.random.choice(new_width - width + 1)
    endx = startx + width
    image = image[:, startx:endx]
  elif width > new_width:
    startx = np.random.choice(width - new_width + 1)
    endx = startx + new_width
    image2 = np.zeros((image.shape[0], width, channel), dtype=image.dtype)
    image2[:, startx:endx] = image
    image = image2

  assert image.shape == (height, width, channel)
  data['image'] = image[..., :-1].astype(image_dtype)
  data['mask'] = image[..., -1].astype(mask_dtype)
  return data

def random_select_position2(image_src, mask_src, image_dst, mask_dst, cell_idx, max_num=100):
  ys, xs = np.nonzero(mask_src == cell_idx)
  center_x1 = int(round(np.mean(xs)))
  center_y1 = int(round(np.mean(ys)))

  blank_ys, blank_xs = np.nonzero(mask_dst == 0)
  x_max, x_min, y_max, y_min = xs.max(), xs.min(), ys.max(), ys.min()
  x_max, x_min = image_dst.shape[1] - (x_max - center_x1), center_x1 - x_min
  y_max, y_min = image_dst.shape[0] - (y_max - center_y1), center_y1 - y_min
  idxs = (blank_xs >= x_min) & (blank_xs <= x_max) & (blank_ys >= y_min) & (blank_ys <= y_max)
  if idxs.sum() == 0: idxs[0] = True
  blank_ys, blank_xs = blank_ys[idxs], blank_xs[idxs]

  blank_idxes = np.arange(len(blank_ys))
  np.random.shuffle(blank_idxes)
  for idx, blank_idx in enumerate(blank_idxes, 1):
    center_x2 = blank_xs[blank_idx]
    center_y2 = blank_ys[blank_idx]
    new_xs = xs - center_x1 + center_x2
    new_ys = ys - center_y1 + center_y2
    image, mask, flags = inner_move2(image_src, mask_src, image_dst, mask_dst, cell_idx, xs, ys, new_xs, new_ys)
    if flags or idx >= max_num:
      break
  return image, mask, flags

def inner_move2(image_src, mask_src, image_dst, mask_dst, cell_idx, xs, ys, new_xs, new_ys, pixels=None, x_limits=None, y_limits=None):
  height, width, channel = image_dst.shape
  if x_limits is None:
    x_limits = (0, width - 1)
  if y_limits is None:
    y_limits = (0, height - 1)
  flags = False
  if np.min(new_xs) > x_limits[0] and np.max(new_xs) < x_limits[1] and np.min(new_ys) > y_limits[0] \
      and np.max(new_ys) < y_limits[1] \
      and len(np.array(list(set(np.unique(mask_dst[new_ys, new_xs]).tolist()) - {0}))) == 0:
    if pixels is None:
      pixels = image_src[ys, xs].copy()
    image_dst[new_ys, new_xs] = pixels
    new_cell_idx = np.max(mask_dst) + 1
    mask_dst[new_ys, new_xs] = new_cell_idx
    flags = True
  return image_dst, mask_dst, flags

def cell_crop_augment(image, mask, paddings=(20, 20, 20, 20)):
  top, bottom, left, right = paddings
  label_image = measure.label(mask)
  max_area = 0
  for region in measure.regionprops(label_image):
    if region.area > max_area:
      max_area = region.area
      min_row, min_col, max_row, max_col = region.bbox

  min_row, min_col = max(min_row - top, 0), max(min_col - left, 0)
  max_row, max_col = min(max_row + bottom, mask.shape[0]), min(max_col + right, mask.shape[1])

  image = image[min_row:max_row, min_col:max_col]
  mask = mask[min_row:max_row, min_col:max_col]
  return image, mask
