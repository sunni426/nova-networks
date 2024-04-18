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

@torch.no_grad()
def evaluate(data_loader, model, device):

    device = 'cuda'
    print('Testing...')
    
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
    print('* mAP {map:.3f} loss {losses.global_avg:.3f}'
          .format(map=map, losses=metric_logger.loss))


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