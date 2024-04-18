# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from util.datasets import a_ordinary_collect_method, custom_collate_fn

import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import torch
# from utils import rand_bbox
# from utils.mix_methods import snapmix, cutmix, cutout, as_cutmix, mixup
# from utils.metric import macro_multilabel_auc
import pickle as pk
from path import Path
from torchinfo import summary
import csv
import os
try:
    from apex import amp
except:
    pass
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import roc_auc_score
from gradcam import *
from scipy import stats

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    mixup_fn = None # Sunni added
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    class_weight = [0.02531037, 0.06579517, 0.04364358, 0.04229549, 0.03539962,
          0.03934447, 0.04536092, 0.03703704, 0.04307305, 0.05735393,
          0.04914732, 0.30151134, 0.0418487 , 0.0347524 , 0.03067138,
          0.10425721, 0.03305898, 0.05933908, 0.15075567]
    pos_weight = torch.tensor(class_weight).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none').to(device)

    # Added, Apr 9 to fix RuntimeError: Input type (torch.cuda.HalfTensor) and weight type (torch.FloatTensor) should be the same
    model.to(device)

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        # print(f'samples: {samples.shape}, targets: {targets.shape}') # samples: torch.Size([32, 3, 224, 224]), targets: torch.Size([32, 1, 19])
        samples = samples.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).float()
        targets = targets.squeeze(1)

        # Don't use this for now, Apr 9
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            # print(f'outputs: {outputs.shape}, targets: {targets.shape}') batch_size x 19
            loss = criterion(outputs, targets)

        loss_value = loss.mean()
        loss = loss.mean() # added

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1) # og, *1000
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):

    device = 'cuda' 
    
    # criterion = torch.nn.CrossEntropyLoss()
    class_weight = [0.02531037, 0.06579517, 0.04364358, 0.04229549, 0.03539962,
          0.03934447, 0.04536092, 0.03703704, 0.04307305, 0.05735393,
          0.04914732, 0.30151134, 0.0418487 , 0.0347524 , 0.03067138,
          0.10425721, 0.03305898, 0.05933908, 0.15075567]
    pos_weight = torch.tensor(class_weight)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none').to(device)
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target = target.squeeze(1)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target[:,0], topk=(1, 5)) # OG # only first col of target?? Apr 11. # doesn't work. try map
        map = calc_metrics(predicted=output, truth=target)
        save_csv(predicted=output, truth=target)

        batch_size = images.shape[0]
        # metric_logger.update(loss=loss.item()) # OG, doesn't work for BCE
        metric_logger.update(loss = loss.mean()) # BCE loss size: [32,19]
        metric_logger.meters['map'].update(map, n=batch_size) # ADDED
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print('* mAP {map:.3f} loss {losses.global_avg:.3f}'
          .format(map=map, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def calc_metrics(predicted, truth):

    predicted = np.array(predicted.cpu())
    truth = np.array(truth.cpu())
    roc_values = []
    mAP = []
    data_len = len(truth)
    mAP_img = mean_average_precision(truth, predicted)
    mAP.append(mAP_img)
    mAP = sum(mAP)/data_len

    return mAP


def average_precision(y_true, y_pred):
    """
    Calculate average precision for a single sample.

    Parameters:
        y_true (1D array): True labels.
        y_pred (1D array): Predicted labels.

    Returns:
        float: Average Precision (AP) for the sample.
    """
    num_true = np.sum(y_true)
    if num_true == 0:
        return 0.0

    sorted_indices = np.argsort(y_pred)[::-1]
    precision = 0.0
    num_correct = 0.0

    for i, idx in enumerate(sorted_indices):
        if round(y_true[idx],8) == 0.90526316: # correct '1'
            num_correct += 1
            precision += num_correct / (i + 1)
            # print('CORRECT!')

    return precision / num_true


def mean_average_precision(y_true, y_pred):
    """
    Calculate Mean Average Precision (mAP) for a multilabel setting.

    Parameters:
        y_true (2D array): True labels.
        y_pred (2D array): Predicted labels.

    Returns:
        float: Mean Average Precision (mAP) across all samples.
    """
    num_samples, num_labels = y_true.shape
    total_ap = 0.0

    for i in range(num_samples):
        total_ap += average_precision(y_true[i], y_pred[i])

    return total_ap / num_samples


def precision_recall_curve(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]

        precision = precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        recall = recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls


def save_csv(predicted, truth):

    predicted = np.array(predicted.cpu())
    truth = np.array(truth.cpu())
    predicted = np.round(predicted, decimals=4)
    truth = np.round(truth, decimals=4)
    data_len = len(truth)
    
    # path = Path(os.path.dirname(os.path.realpath(__file__)))
    # csv_file_path = f'{path}/dataloaders/split/{cfg.experiment.csv_valid}'
    # # Open the CSV file and read the first column into a list
    # with open(csv_file_path, 'r') as csvfile:
    #     csv_reader = csv.reader(csvfile)
    #     row_ids = [row[0] for row in csv_reader]
    # row_ids = row_ids[1:]        
    # print(f'row_ids: {row_ids}')
    # predicted = predicted.reshape((len(dl)*n_cell, 19)) # now, num_valid*num_cellsx19 

    # # Add a new column with string IDs at index 0
    predicted_with_ids, truth_with_ids = predicted, truth
    # for img in range(data_len):
    #     start = img*10
    #     end = start + 10
        # predicted_with_ids.append(np.column_stack((np.broadcast_to(np.array(row_ids[img]), (10, 1)).tolist(), predicted[start:end, :])))
        # truth_with_ids.append(np.column_stack((row_ids[img], np.expand_dims(truth[img],0))))

    # Hardcoding IDs for now, create an array of strings ('ID') Apr 14, 2024
    ids = np.array([i for i in range(0, data_len)])  # Assuming 32 rows
    
    # Insert the 'ID' column at the beginning of the array
    predicted_with_ids = np.insert(predicted_with_ids, 0, ids, axis=1)
    predicted_with_ids = np.array(predicted_with_ids).reshape((data_len, 20)) # 20, because 19 labels + ID
    truth_with_ids = np.insert(truth_with_ids, 0, ids, axis=1)
    truth_with_ids = np.array(truth_with_ids).reshape((data_len, 20)) # 20, because 19 labels + ID
    
    # Create header with 'ID' added at the beginning
    header = ','.join(['ID'] + [f'class{i}' for i in range(truth.shape[1])])
    
    # Get the directory and file paths
    # base_path = Path(os.path.dirname(os.path.realpath(__file__))) / 'results' / cfg.basic.id
    # pred_path = base_path / 'pred.csv'
    # truth_path = base_path / 'truth.csv'
    pred_path = 'pred.csv'
    truth_path = 'truth.csv'
    
    # Save with truncated values and the new ID column
    np.savetxt(pred_path, predicted_with_ids, fmt='%s', delimiter=',', header=header, comments='')
    np.savetxt(truth_path, truth_with_ids, fmt='%s', delimiter=',', header=header, comments='')
    
    # print(os.getcwd())

    # # Save predicted array as CSV
    # np.savetxt('predicted.csv', predicted, delimiter=',')
    
    # # Save truth array as CSV
    # np.savetxt('truth.csv', truth, delimiter=',')