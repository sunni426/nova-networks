import numpy as np
import torch.nn as nn
import torch.optim as optim

class SchedulerBase(object):
  def __init__(self):
    self._is_load_best_weight = False
    self._is_load_best_optim = False
    self._lr = 0.01
    self._optimizer = None

  def schedule(self,net, epoch, epochs, **kwargs):
    raise Exception('Did not implemented')

  def is_load_best_weight(self):
    return self._is_load_best_weight

  def is_load_best_optim(self):
    return self._is_load_best_optim

  def reset(self):
    self._is_load_best_weight = False
    self._load_best_optim = False

  def get_optimizer(self):
    return self._optimizer

class Adam1(SchedulerBase):
  def __init__(self, model):
    super(Adam1, self).__init__()
    self.model = model
    self._lr = 3e-4

  def set_optim(self, model):
    self._optimizer = optim.Adam(model.parameters(), lr=self._lr)

  def schedule(self, epoch, epochs, **kwargs):
    lr = 30e-5
    if epoch > 25:
      lr = 15e-5
    if epoch > 35:
      lr = 7.5e-5
    if epoch > 45:
      lr = 3e-5
    if epoch > 50:
      lr = 1e-5
    for param_group in self._optimizer.param_groups:
      param_group['lr'] = lr
    self._lr = self._optimizer.param_groups[0]['lr']
    return self._optimizer, self._lr

class Adamft(SchedulerBase):
  def __init__(self, model):
    super(Adamft, self).__init__()
    self.model = model
    self._lr = 1e-5

  def set_optim(self, model):
    self._optimizer = optim.Adam(model.parameters(), lr=self._lr)

  def schedule(self, epoch, epochs, **kwargs):
    lr = 1e-5
    for param_group in self._optimizer.param_groups:
      param_group['lr'] = lr
    self._lr = self._optimizer.param_groups[0]['lr']
    return self._optimizer, self._lr
