import sys
sys.path.insert(0, '..')
import cv2

from config.config import *

def load_rgby(image_dir, image_id, suffix='jpg', in_channels=4):
  images = [
    cv2.imread(f'{image_dir}/{image_id}_{color}.{suffix}', cv2.IMREAD_GRAYSCALE) for color in COLORS[0:in_channels]
  ]
  for image in images:
    if image is None:
      return None
  rgby = np.stack(images, axis=-1)
  return rgby

def rgby2rgb(rgby, data_type='rgby'):
  for dtype in data_type:
    assert dtype in 'rgby'

  rgb = []
  for idx, dtype in enumerate('rgb', 0):
    if dtype in data_type:
      rgb.append(rgby[..., [idx]])
    else:
      rgb.append(np.zeros_like(rgby[..., [idx]]))
  rgb = np.concatenate(rgb, axis=2)

  if 'y' in data_type:
    rgb = rgb.astype('float32')
    rgb[..., 0] += rgby[..., -1] / 2
    rgb[..., 1] += rgby[..., -1] / 2
    rgb = rgb / rgb.max() * 255
    rgb = rgb.astype('uint8')

  return rgb

def load_mask(cellmask_dir, image_id, masktype='cellmask'):
  mask = cv2.imread(f'{cellmask_dir}/{image_id}_{masktype}.png', flags=cv2.IMREAD_GRAYSCALE)
  return mask

def generate_cell_indices(cell_mask):
  cell_indices = np.sort(list(set(np.unique(cell_mask).tolist()) - {0, }))
  return cell_indices

def mask2image(image, mask, dx=1, color=(255, 255, 255)):
  mask = np.concatenate([
    np.zeros((dx, mask.shape[1]), dtype=mask.dtype),
    mask,
    np.zeros((dx, mask.shape[1]), dtype=mask.dtype),
  ], axis=0)
  mask = np.concatenate([
    np.zeros((mask.shape[0], dx), dtype=mask.dtype),
    mask,
    np.zeros((mask.shape[0], dx), dtype=mask.dtype),
  ], axis=1)

  color = np.array(list(color))
  image[mask[dx:-dx, 2 * dx:] != mask[dx:-dx, :-2*dx]] = color
  image[mask[2 * dx:, dx:-dx] != mask[:-2*dx, dx:-dx]] = color
  return image
