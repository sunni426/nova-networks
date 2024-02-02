import sys
sys.path.insert(0, '..')
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import DataParallel, Parameter
from torch.nn.parallel import DistributedDataParallel
from collections.abc import Iterable

from config.config import *

def remove_prefix(load_state_dict, name):
  new_load_state_dict = dict()
  for key in load_state_dict.keys():
    if key.startswith(name):
      dst_key = key.replace(name, '')
    else:
      dst_key = key
    new_load_state_dict[dst_key] = load_state_dict[key]
  load_state_dict = new_load_state_dict
  return load_state_dict

# load_pretrained ------------------------------------
def load_pretrained_state_dict(net, load_state_dict, strict=False, can_print=True):
  if 'epoch' in load_state_dict and can_print:
    epoch = load_state_dict['epoch']
    print(f'load epoch:{epoch:.2f}')
  if 'state_dict' in load_state_dict:
    load_state_dict = load_state_dict['state_dict']
  elif 'model_state_dict' in load_state_dict:
    load_state_dict = load_state_dict['model_state_dict']
  elif 'model' in load_state_dict:
    load_state_dict = load_state_dict['model']
  if isinstance(net, (DataParallel, DistributedDataParallel)):
    state_dict = net.module.state_dict()
  else:
    state_dict = net.state_dict()

  load_state_dict = remove_prefix(load_state_dict, 'module.')
  load_state_dict = remove_prefix(load_state_dict, 'base_model.')

  for key in list(load_state_dict.keys()):
    if key not in state_dict:
      if strict:
        raise Exception(f'not in {key}')
      if can_print:
        print('not in', key)
      continue
    if load_state_dict[key].size() != state_dict[key].size():
      if strict or (len(load_state_dict[key].size()) != len(state_dict[key].size())):
        raise Exception(f'size not the same {key}: {load_state_dict[key].size()} -> {state_dict[key].size()}')
      if can_print:
        print(f'{key} {load_state_dict[key].size()} -> {state_dict[key].size()}')
      state_slice = [slice(s) for s in np.minimum(np.array(load_state_dict[key].size()), np.array(state_dict[key].size()))]
      state_dict[key][state_slice] = load_state_dict[key][state_slice]
      continue
    state_dict[key] = load_state_dict[key]

  if isinstance(net, (DataParallel, DistributedDataParallel)):
    net.module.load_state_dict(state_dict)
  else:
    net.load_state_dict(state_dict)
  return net

def load_pretrained(net, pretrained_file, strict=False, can_print=False):
  if can_print:
    print(f'load pretrained file: {pretrained_file}')
  load_state_dict = torch.load(pretrained_file, map_location=torch.device('cpu'))
  net = load_pretrained_state_dict(net, load_state_dict, strict=strict, can_print=can_print)
  return net

def load_pretrained_maskrcnn(net, pretrained_file, strict=False, can_print=False):
  if can_print:
    print(f'load pretrained file: {pretrained_file}')
  load_state_dict = torch.load(pretrained_file, map_location=torch.device('cpu'))
  load_state_dict['state_dict'] = remove_prefix(load_state_dict['state_dict'], 'backbone.')
  net = load_pretrained_state_dict(net, load_state_dict, strict=strict, can_print=can_print)
  return net

# freeze ------------------------------------
def set_freeze_by_names(model, layer_names, freeze=True):
  if not isinstance(layer_names, Iterable):
    layer_names = [layer_names]

  layer_names_map = {}
  for layer_name in layer_names:
    layer_name_list = layer_name.split('.')
    sub_map1 = layer_names_map
    for s in layer_name_list[:-1]:
      sub_map2 = sub_map1.get(s, {})
      sub_map1[s] = sub_map2
      sub_map1 = sub_map2
    sub_map1[layer_name_list[-1]] = True

  for name, param in model.named_parameters():
    name_list = name.split('.')
    sub_map = layer_names_map
    for sub_name in name_list:
      if sub_name not in sub_map:
        break
      if isinstance(sub_map[sub_name], dict):
        sub_map = sub_map[sub_name]
        continue
      assert sub_map[sub_name] == True
      param.requires_grad = not freeze

def freeze_by_names(model, layer_names):
  set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
  set_freeze_by_names(model, layer_names, False)

def set_freeze_bn_by_names(model, layer_names, suffix=None):
  if not isinstance(layer_names, Iterable):
    layer_names = [layer_names]

  layer_names_map = {}
  if suffix is not None:
    sub_layer_names_map = {}
    layer_names_map[suffix] = sub_layer_names_map
  else:
    sub_layer_names_map = layer_names_map
  for layer_name in layer_names:
    layer_name_list = layer_name.split('.')
    sub_map1 = sub_layer_names_map
    for s in layer_name_list[:-1]:
      sub_map2 = sub_map1.get(s, {})
      sub_map1[s] = sub_map2
      sub_map1 = sub_map2
    sub_map1[layer_name_list[-1]] = True

  for name, m in model.named_modules():
    classname = m.__class__.__name__
    if (classname.find('BatchNorm') == -1):
      continue
    name_list = name.split('.')
    sub_map = layer_names_map
    for sub_name in name_list:
      if sub_name not in sub_map:
        break
      if isinstance(sub_map[sub_name], dict):
        sub_map = sub_map[sub_name]
        continue
      assert sub_map[sub_name] == True

      m.eval()
      m.weight.requires_grad = False
      m.bias.requires_grad = False

def freeze_bn_by_names(model, layer_names, suffix=None):
  set_freeze_bn_by_names(model, layer_names, suffix=suffix)

class ArcMarginProduct(nn.Module):
  r"""Implement of large margin arc distance: :
    Args:
      in_features: size of each input sample
      out_features: size of each output sample
      s: norm of input feature
      m: margin
      cos(theta + m)
    """

  def __init__(self, in_features, out_features):
    super(ArcMarginProduct, self).__init__()
    self.weight = Parameter(torch.FloatTensor(out_features, in_features))
    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)

  def forward(self, features):
    cosine = F.linear(F.normalize(features), F.normalize(self.weight))
    return cosine

# https://www.kaggle.com/debarshichanda/seresnext50-but-with-attention
def convert_act_cls(model, layer_type_old, layer_type_new):
  for name, module in reversed(model._modules.items()):
    if len(list(module.children())) > 0:
      # recurse
      model._modules[name] = convert_act_cls(module, layer_type_old, layer_type_new)
    if type(module) == layer_type_old:
      model._modules[name] = layer_type_new
  return model

class CBAM_Module(nn.Module):
  def __init__(self, channels, reduction):
    super(CBAM_Module, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)
    self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,padding=0)
    self.relu = nn.ReLU(inplace=True)
    self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,padding=0)
    self.sigmoid_channel = nn.Sigmoid()
    self.conv_after_concat = nn.Conv2d(2, 1,
                                       kernel_size = 3,
                                       stride=1,
                                       padding = 3//2)
    self.sigmoid_spatial = nn.Sigmoid()

  def forward(self, x):
    # Channel attention module
    module_input = x
    avg = self.avg_pool(x)
    mx = self.max_pool(x)
    avg = self.fc1(avg)
    mx = self.fc1(mx)
    avg = self.relu(avg)
    mx = self.relu(mx)
    avg = self.fc2(avg)
    mx = self.fc2(mx)
    x = avg + mx
    x = self.sigmoid_channel(x)
    # Spatial attention module
    x = module_input * x
    module_input = x
    # b, c, h, w = x.size()
    avg = torch.mean(x, 1, True)
    mx, _ = torch.max(x, 1, True)
    x = torch.cat((avg, mx), 1)
    x = self.conv_after_concat(x)
    att = self.sigmoid_spatial(x)
    ret = module_input * att
    return ret,att

class SSELayer(nn.Module):
  def __init__(self, channel_in, multiply,
               attention_kernel_size=1,position_encode=False):
    super(SSELayer, self).__init__()
    c = channel_in
    C = 1
    self.multiply = multiply
    self.conv_in = nn.Conv2d(c, C, kernel_size=attention_kernel_size,padding=attention_kernel_size//2)
    self.bn1 = nn.BatchNorm2d(C)
    self.sigmoid = nn.Sigmoid()
    self.position_encode=position_encode
    self.position_encoded = None
  def forward(self, x):
    b, c, _, _ = x.size()
    y = self.conv_in(x)
    y = self.bn1(y)
    y = self.sigmoid(y)
    if self.multiply == True:
      return x * y
    else:
      return y
