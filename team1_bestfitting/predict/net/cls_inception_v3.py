import sys
sys.path.insert(0, '..')
import torch
import torch.nn as nn

from config.config import *
from utilities.model_util import *
from net.backbone.inception_v3 import *
from utilities.model_util import load_pretrained

model_names = {
  'inception_v3': 'inception_v3_google-1a9a5a14.pth',
}

## net  ######################################################################
class InceptionV3(nn.Module):

  def __init__(self, args,
               att_type=None,
               feature_net='inception_v3', do_ml=False
               ):
    super().__init__()
    self.args = args
    self.att_type = att_type
    self.do_ml = do_ml

    self.backbone = inception_v3()
    if args.pretrained:
      self.backbone = load_pretrained(self.backbone, f'{DIR_CFGS.PRETRAINED_DIR}/{model_names[feature_net]}',
                                      strict=False, can_print=args.can_print)

    if args.in_channels > 3:
      w = self.backbone.Conv2d_1a_3x3.conv.weight
      self.backbone.Conv2d_1a_3x3.conv = nn.Conv2d(args.in_channels, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
      self.backbone.Conv2d_1a_3x3.conv.weight = torch.nn.Parameter(
        torch.cat([w] * int(self.args.in_channels // 3) + [w[:, :int(self.args.in_channels % 3), :, :]], dim=1)
      )

    self.backbone.layer0 = nn.Sequential(
      self.backbone.Conv2d_1a_3x3,
      self.backbone.Conv2d_2a_3x3,
      self.backbone.Conv2d_2b_3x3,
    )
    self.backbone.layer1 = nn.Sequential(
      self.backbone.Conv2d_3b_1x1,
      self.backbone.Conv2d_4a_3x3,
    )
    self.backbone.layer2 = nn.Sequential(
      self.backbone.Mixed_5b,
      self.backbone.Mixed_5c,
      self.backbone.Mixed_5d,
    )
    self.backbone.layer3 = nn.Sequential(
      self.backbone.Mixed_6a,
      self.backbone.Mixed_6b,
      self.backbone.Mixed_6c,
      self.backbone.Mixed_6d,
      self.backbone.Mixed_6e,
    )
    self.backbone.layer4 = nn.Sequential(
      self.backbone.Mixed_7a,
      self.backbone.Mixed_7b,
      self.backbone.Mixed_7c,
    )

    feature_dim = 2048
    if self.att_type in ['cbam']:
      self.att_module = CBAM_Module(channels=feature_dim, reduction=32)
    else:
      self.att_module = None

    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.maxpool = nn.AdaptiveMaxPool2d(1)

    feature_nums = 2 * feature_dim
    self.fc_layers = nn.Sequential(
      nn.BatchNorm1d(feature_nums),
      nn.Dropout(p=0.5),
      nn.Linear(feature_nums, feature_dim),
      nn.ReLU(),
      nn.BatchNorm1d(feature_dim),
      nn.Dropout(p=0.5),
    )
    self.logit = nn.Linear(in_features=feature_dim, out_features=args.num_classes)
    if self.do_ml:
      self.ml_logit = ArcMarginProduct(feature_dim, self.args.ml_num_classes)

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
    x = self.backbone.layer0(data['image'])
    x = self.backbone.layer1(x)
    x = self.backbone.layer2(x)
    x = self.backbone.layer3(x)
    x = self.backbone.layer4(x)

    features = x
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
      ml_logits = self.ml_logit(x)
      data['ml_logits'] = ml_logits
    return data

def cls_inception_v3_cbam(args):
  model = InceptionV3(args, feature_net='inception_v3', att_type='cbam')
  return model

def get_model(args, **kwargs):
  net = eval(args.model_name)(args)
  scheduler = None if args.scheduler is None else eval(args.scheduler)(net)
  loss = None if args.loss is None else eval(args.loss)()
  return net, scheduler, loss
