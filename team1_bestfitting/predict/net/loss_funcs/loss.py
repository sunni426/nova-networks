import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import *
from net.loss_funcs.lovasz_losses import lovasz_hinge
from net.loss_funcs.hard_example import get_hard_samples

class FocalLoss(nn.Module):
  def __init__(self, gamma=2):
    super().__init__()
    self.gamma = gamma

  def forward(self, data, epoch=0):
    logits = data['logits']
    labels = data['labels']

    labels = labels.float()
    max_val = (-logits).clamp(min=0)
    loss = logits - logits * labels + max_val + ((-max_val).exp() + (-logits - max_val).exp()).log()

    invprobs = F.logsigmoid(-logits * (labels * 2.0 - 1.0))
    loss = (invprobs * self.gamma).exp() * loss
    if len(loss.size()) == 2:
      loss = loss.sum(dim=1)
    return loss.mean()

class SymmetricLovaszLoss(nn.Module):
  def __init__(self):
    super(SymmetricLovaszLoss, self).__init__()

  def forward(self, data, epoch=0):
    logits = data['logits']
    labels = data['labels']

    labels = labels.float()
    loss = (lovasz_hinge(logits, labels) + lovasz_hinge(-logits, 1 - labels)) / 2
    return loss

class HardLogLoss(nn.Module):
  def __init__(self):
    super(HardLogLoss, self).__init__()
    self.bce_loss = nn.BCEWithLogitsLoss()

  def forward(self, data, epoch=0):
    logits = data['logits']
    labels = data['labels']

    labels = labels.float()
    loss = 0
    for i in range(logits.shape[1]):
      logit_ac = logits[:, i]
      label_ac = labels[:, i]
      logit_ac, label_ac = get_hard_samples(logit_ac, label_ac)
      loss += self.bce_loss(logit_ac, label_ac)
    loss = loss / logits.shape[1]
    return loss

class FocalSymmetricLovaszHardLogLoss(nn.Module):
  def __init__(self):
    super(FocalSymmetricLovaszHardLogLoss, self).__init__()
    self.focal_loss = FocalLoss()
    self.slov_loss = SymmetricLovaszLoss()
    self.log_loss = HardLogLoss()

  def forward(self, data, epoch=0):
    focal_loss = self.focal_loss(data, epoch=epoch)
    slov_loss = self.slov_loss(data, epoch=epoch)
    log_loss = self.log_loss(data, epoch=epoch)
    loss = focal_loss * 0.5 + slov_loss * 0.5 + log_loss * 0.5
    return loss

def L2_Loss(A_tensors, B_tensors):
  return torch.pow(A_tensors - B_tensors, 2)

class PuzzleLossV2a(nn.Module):
  def __init__(self):
    super().__init__()
    self.class_loss = FocalSymmetricLovaszHardLogLoss()
    self.re_loss = L2_Loss

  def forward(self, data, epoch=0):
    class_loss = self.class_loss({'logits': data['logits'], 'labels': data['labels']})
    p_class_loss = self.class_loss({'logits': data['p_logits'], 'labels': data['labels']})
    re_loss = self.re_loss(data['features'], data['p_features']).mean()
    return class_loss + p_class_loss + re_loss

class ArcFaceLoss(nn.modules.Module):
  def __init__(self, s=30.0, m=0.5):
    super(ArcFaceLoss, self).__init__()
    self.classify_loss = nn.CrossEntropyLoss()
    self.s = s
    self.easy_margin = False
    self.cos_m = math.cos(m)
    self.sin_m = math.sin(m)
    self.th = math.cos(math.pi - m)
    self.mm = math.sin(math.pi - m) * m

  def forward(self, data, epoch=0):
    cosine, labels = data['ml_logits'], data['ml_labels']
    indices = ~torch.isnan(labels)
    cosine = cosine[indices]
    labels = labels[indices].long()

    if len(labels) == 0:
      loss = cosine.sum()  # zero
    else:
      sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
      phi = cosine * self.cos_m - sine * self.sin_m
      if self.easy_margin:
        phi = torch.where(cosine > 0, phi, cosine)
      else:
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

      one_hot = torch.zeros(cosine.size(), device=labels.device)
      one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
      # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
      output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
      output *= self.s
      loss1 = self.classify_loss(output, labels)
      loss2 = self.classify_loss(cosine, labels)
      loss = (loss1 + loss2) / 2
    return loss

class PuzzleV2aArcFaceLossV2(nn.modules.Module):
  def __init__(self):
    super(PuzzleV2aArcFaceLossV2, self).__init__()
    self.loss1 = PuzzleLossV2a()
    self.loss2 = ArcFaceLoss()

  def forward(self, data, epoch=0):
    loss1 = self.loss1(data, epoch=epoch)

    loss21 = self.loss2({'ml_logits': data['ml_logits'], 'ml_labels': data['ml_labels']}, epoch=epoch)
    loss22 = self.loss2({'ml_logits': data['p_ml_logits'], 'ml_labels': data['ml_labels']}, epoch=epoch)
    loss2 = (loss21 + loss22) / 2

    loss = loss1 + loss2
    return loss

class FocalLossV1(nn.Module):
  def __init__(self, gamma=2):
    super().__init__()
    self.gamma = gamma

  def forward(self, data, epoch=0):
    logits = data['logits']
    labels = data['labels']
    indices = labels >= 0
    logits = logits[indices]
    labels = labels[indices]

    labels = labels.float()
    max_val = (-logits).clamp(min=0)
    loss = logits - logits * labels + max_val + ((-max_val).exp() + (-logits - max_val).exp()).log()

    invprobs = F.logsigmoid(-logits * (labels * 2.0 - 1.0))
    loss = (invprobs * self.gamma).exp() * loss
    if len(loss.size()) == 2:
      loss = loss.sum(dim=1)
    return loss.mean()
