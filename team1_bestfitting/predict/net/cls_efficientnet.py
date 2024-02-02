import sys
sys.path.insert(0, '..')
import torch.nn as nn

from config.config import *
from net.scheduler import *
from utilities.model_util import *
from net.loss_funcs.loss import *
from net.backbone.efficientnet import *
from net.backbone.efficientnet_utils import efficientnet_params

## net  ######################################################################
class EfficientnetClsNet(nn.Module):

  def __init__(self, args, feature_net='efficientnet-b5', do_ml=False, do_plate_ml=False, att_type=None):
    super().__init__()
    self.args = args
    self.feature_net = feature_net
    self.do_ml = do_ml
    self.do_plate_ml = do_plate_ml
    self.att_type = att_type

    if self.args.pretrained:
      self.backbone = EfficientNet.from_pretrained(feature_net, model_dir=DIR_CFGS.PRETRAINED_DIR, can_print=args.can_print)
    else:
      self.backbone = EfficientNet.from_name(feature_net, override_params={'num_classes': 1000})
    Conv2d = get_same_padding_conv2d(image_size=efficientnet_params(feature_net)[2])

    w = self.backbone._conv_stem.weight
    self.backbone._conv_stem = Conv2d(self.args.in_channels, self.backbone._conv_stem.out_channels, kernel_size=3, stride=2, bias=False)
    self.backbone._conv_stem.weight = torch.nn.Parameter(
      torch.cat([w] * int(self.args.in_channels // 3) + [w[:, :int(self.args.in_channels % 3), :, :]], dim=1)
    )

    att_in_channels = self.backbone._conv_head.out_channels
    if self.att_type in ['cbam']:
      self.att_module = CBAM_Module(channels=att_in_channels, reduction=32)
    else:
      self.att_module = None

    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.maxpool = nn.AdaptiveMaxPool2d(1)

    in_channels = self.backbone._conv_head.out_channels * 2
    self.fc_layers = nn.Sequential(
      nn.BatchNorm1d(in_channels),
      nn.Dropout(p=0.5),
      nn.Linear(in_channels, 512),
      nn.ReLU(),
      nn.BatchNorm1d(512),
      nn.Dropout(p=0.5),
    )

    self.logit = nn.Linear(in_features=512, out_features=self.args.num_classes)
    if self.do_ml:
      self.ml_logit = ArcMarginProduct(512, self.args.ml_num_classes)

  def gap_forward(self, x):
    x = torch.cat([self.avgpool(x), self.maxpool(x)], dim=1)
    x = x.view(x.size(0), -1)
    x = self.fc_layers(x)

    p_logits = self.logit(x)
    data = {'p_logits': p_logits}
    if self.do_ml:
      p_ml_logits = self.ml_logit(x)
      data['p_ml_logits'] = p_ml_logits
    return data

  def forward(self, data, **kwargs):
    features = x = self.backbone.extract_features(data['image'])

    if self.att_type in ['cbam']:
      x, _ = self.att_module(x)
    if x.size() == features.size():
      features = x

    x = torch.cat([self.avgpool(x), self.maxpool(x)], dim=1)
    x = x.view(x.size(0), -1)
    x = self.fc_layers(x)

    logits = self.logit(x)
    data['logits'] = logits
    data['features'] = features
    data['feature_vector'] = x
    if self.do_ml:
      ml_features = x
      ml_logits = self.ml_logit(x)
      data['ml_logits'] = ml_logits
      data['ml_features'] = ml_features

    return data

def cls_efficientnet_b0_cbam(args):
  model = EfficientnetClsNet(args, feature_net='efficientnet-b0', att_type='cbam')
  return model

def ml_efficientnet_b0_cbam(args):
  model = EfficientnetClsNet(args, feature_net='efficientnet-b0', do_ml=True, att_type='cbam')
  return model

def get_model(args, **kwargs):
  net = eval(args.model_name)(args)
  scheduler = None if args.scheduler is None else eval(args.scheduler)(net)
  loss = None if args.loss is None else eval(args.loss)()
  return net, scheduler, loss
