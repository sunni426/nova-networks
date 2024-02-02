import sys
sys.path.insert(0, '..')
import argparse
import pandas as pd
from tqdm import tqdm
from timeit import default_timer as timer
from importlib import import_module

import torch
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from utilities.augment_util import *
from utilities.model_util import load_pretrained
from dataset.cell_cls_dataset import CellClsDataset as HPA2021Dataset
from dataset.cell_cls_dataset import cls_collate_fn as collate_fn
cudnn.benchmark = True

def initialize_environment(args):
  seed = 100
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  # COMMON
  args.device = 'cuda' if args.gpus else 'cpu'
  args.can_print = True
  args.seed = seed
  args.debug = False

  # MODEL
  args.scheduler = None
  args.loss = None
  args.pretrained = False

  # DATASET
  args.image_size = args.image_size.split(',')
  if len(args.image_size) == 1:
    args.image_size = [int(args.image_size[0]), int(args.image_size[0])]
  elif len(args.image_size) == 2:
    args.image_size = [int(args.image_size[0]), int(args.image_size[1])]
  else:
    raise ValueError(','.join(args.image_size))

  args.num_workers = 4
  args.split_type = 'random'
  args.suffix = 'png'
  args.augments = args.augments.split(',')

  if args.can_print:
    if args.gpus:
      print(f'use gpus: {args.gpus}')
    else:
      print(f'use cpu')
  if args.gpus:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
  args.model_fpath = f'{DIR_CFGS.MODEL_DIR}/{args.model_dir}/fold{args.fold}/{args.model_epoch}.pth'
  args.output_dir = f'{DIR_CFGS.RESULT_DIR}/{args.model_dir}/fold{args.fold}/epoch_{args.model_epoch}'
  args.feature_dir = f'{DIR_CFGS.FEATURE_DIR}/{args.model_dir}/fold{args.fold}/epoch_{args.model_epoch}'

  if args.can_print:
    print(f'output dir: {args.output_dir}')
    print(f'feature dir: {args.feature_dir}')
  os.makedirs(args.output_dir, exist_ok=True)
  os.makedirs(args.feature_dir, exist_ok=True)
  return args

def load_model(args):
  image_size = args.image_size
  args.image_size = image_size[0]
  model = import_module(f'net.{args.module}').get_model(args)[0]
  args.image_size = image_size
  model = load_pretrained(model, args.model_fpath, strict=True, can_print=args.can_print) #ERROR HERE loading models
  model = model.eval().to(args.device)
  if args.device == 'cuda':
    model = DataParallel(model)
  return model

def generate_dataloader(args):
  test_dataset = HPA2021Dataset(
    args,
    transform=None,
    dataset=args.dataset,
    is_training=False,
  )
  test_dataset.set_part(part_start=args.part_start, part_end=args.part_end)
  _ = test_dataset[0]
  test_loader = DataLoader(
    test_dataset,
    sampler=SequentialSampler(test_dataset),
    batch_size=args.batch_size,
    drop_last=False,
    num_workers=args.num_workers,
    pin_memory=False,
    collate_fn=collate_fn,
  )
  print(f'num: {test_dataset.__len__()}')
  return test_loader

def predict_cell_result_augment (args, test_loader, model, augment='default'):
  result_ids = []
  result_cell_masks = []
  result_cell_maskids = []
  result_probs = []
  result_feats = []
  test_loader.dataset.transform = eval(f'cls_augment_{augment}')
  with torch.no_grad():
    for it, iter_data in tqdm(enumerate(test_loader, 0), total=len(test_loader), desc=f'cell {augment}'):
      iter_data['image'] = Variable(iter_data['image'].to(args.device))
      data = {'image': iter_data['image']}
      outputs = model(data)
      logits = outputs['logits']
      probs = torch.sigmoid(logits)
      probs = probs.to('cpu').detach().numpy()
      feats = outputs['feature_vector'].to('cpu').detach().numpy()

      result_ids.extend(iter_data[ID])
      result_cell_masks.extend(iter_data['cellmask'])
      result_cell_maskids.extend(iter_data['maskid'])
      result_probs.extend(probs)
      result_feats.extend(feats)
  test_loader.dataset.transform = None

  result_feats = np.array(result_feats)
  result_probs = np.array(result_probs)
  prob_cols = ALIASES if args.label is None else [ALIASES[args.label]]
  result_df = pd.DataFrame(data=result_probs, columns=prob_cols)
  result_df.insert(0, ID, result_ids)
  result_df.insert(1, 'mask', result_cell_masks)
  result_df.insert(2, 'maskid', result_cell_maskids)
  return result_df, result_feats

def predict_cell_result(args, test_loader, model):
  result_df = None
  prob_cols = ALIASES if args.label is None else [ALIASES[args.label]]
  for augment in args.augments:
    if (args.part_end - args.part_start) == 1:
      augment_result_fpath = f'{args.output_dir}/cell_result_{args.dataset}_{augment}.csv'
    else:
      augment_result_fpath = f'{args.output_dir}/cell_result_{args.dataset}_{augment}_{args.part_start:.2f}_{args.part_end:.2f}.csv'
    if args.cell_type:
      augment_result_fpath = augment_result_fpath.replace('.csv', f'_cell_v{args.cell_type}.csv')
    if ope(augment_result_fpath) and not args.overwrite:
      print(f'load augment_result: {augment_result_fpath}')
      augment_result_df = pd.read_csv(augment_result_fpath)
    else:
      model = load_model(args) if model is None else model #ERROR HERE
      test_loader = generate_dataloader(args) if test_loader is None else test_loader
      augment_result_df, result_feats = predict_cell_result_augment(args, test_loader, model, augment=augment)
      augment_result_df.to_csv(augment_result_fpath, index=False, encoding='utf-8')
      augment_feature_fpath = augment_result_fpath.replace('.csv', '').replace('result', 'features')
      np.savez_compressed(augment_feature_fpath, feats=result_feats)
      print(f'Saving features at: {augment_feature_fpath}')

    if result_df is None:
      result_df = augment_result_df
    else:
      assert np.array_equal(result_df[ID], augment_result_df[ID])
      result_df[prob_cols] = result_df[prob_cols].values + augment_result_df[prob_cols].values
  result_df[prob_cols] = result_df[prob_cols] / len(args.augments)

  if (args.part_end - args.part_start) == 1:
    result_fpath = f'{args.output_dir}/cell_result_{args.dataset}.csv'
  else:
    result_fpath = f'{args.output_dir}/cell_result_{args.dataset}_{args.part_start:.2f}_{args.part_end:.2f}.csv'
  if args.cell_type:
    result_fpath = result_fpath.replace('.csv', f'_cell_v{args.cell_type}.csv')
  print(f'result_fpath: {result_fpath}')
  result_df.to_csv(result_fpath, index=False, encoding='utf-8')
  print(result_df.head())
  return result_df

def main(args):
  start_time = timer()
  args = initialize_environment(args)
  model = None
  test_loader = None

  result_df = predict_cell_result(args, test_loader, model) #ERROR HERE
  end_time = timer()
  print(f'time: {(end_time - start_time) / 60.:.2f} min.')
  return result_df

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='PyTorch Classification')
  parser.add_argument('--module', type=str, default='cls_efficientnet', help='model')
  parser.add_argument('--model_name', type=str, default='cls_efficientnet_b3', help='model_name')
  parser.add_argument('--model_dir', type=str, default=None, help='model_dir')
  parser.add_argument('--model_epoch', type=str, default='99.00_ema', help='model_epoch')
  parser.add_argument('--gpus', default=None, type=str, help='use gpus')
  parser.add_argument('--image_size', default='512', type=str, help='image_size')
  parser.add_argument('--batch_size', default=2, type=int, help='batch_size')
  parser.add_argument('--dataset', default='valid', type=str, help='dataset')
  parser.add_argument('--fold', default=0, type=int, help='index of fold')
  parser.add_argument('--augments', default='default', type=str, help='augments')
  parser.add_argument('--part_start', default=0., type=float, help='part_start')
  parser.add_argument('--part_end', default=1., type=float, help='part_end')
  parser.add_argument('--overwrite', default=0, type=int, help='overwrite')
  parser.add_argument('--ml_num_classes', default=ML_NUM_CLASSES, type=int)
  parser.add_argument('--label', type=int, default=None)
  parser.add_argument('--num_classes', default=NUM_CLASSES, type=int)
  parser.add_argument('--cell_type', type=int, default=0)
  parser.add_argument('--in_channels', type=int, default=4)
  parser.add_argument('--kaggle', type=int, default=0)
  parser.add_argument('--cell_complete', type=int, default=0)
  args = parser.parse_args()
  main(args)
