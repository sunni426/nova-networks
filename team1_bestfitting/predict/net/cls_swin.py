import sys
sys.path.insert(0, '..')
import torch
import torch.nn as nn

from config.config import *
from net.scheduler import *
from utilities.model_util import *
from net.loss_funcs.loss import *
from net.backbone.swin_transformer2 import *
from utilities.model_util import load_pretrained,load_pretrained_maskrcnn

model_names = {
  'swin_small_patch4_window7': 'swin_small_patch4_window7_224.pth',
}
## net  ######################################################################
class SwinCls(nn.Module):
  def __init__(self, args, feature_net='swin_small_patch4_window7', do_ml=False, att_type=None):
    super().__init__()
    self.args = args
    self.feature_net = feature_net
    self.do_ml = do_ml
    self.att_type = att_type
    self.backbone, model_args = eval(self.feature_net)()
    if args.pretrained:
      if feature_net.endswith('maskrcnn'):
        self.backbone = load_pretrained_maskrcnn(
          self.backbone, f'{DIR_CFGS.PRETRAINED_DIR}/{model_names[feature_net]}', can_print=args.can_print
        )
      else:
        self.backbone = load_pretrained(
          self.backbone, f'{DIR_CFGS.PRETRAINED_DIR}/{model_names[feature_net]}', can_print=args.can_print
        )

    w = self.backbone.patch_embed.proj.weight
    self.backbone.patch_embed.proj = nn.Conv2d(args.in_channels, model_args['embed_dim'], kernel_size=4, stride=4)
    self.backbone.patch_embed.proj.weight = torch.nn.Parameter(
      torch.cat([w] * int(args.in_channels // 3) + [w[:, :int(args.in_channels % 3), :, :]], dim=1)
    )

    feature_nums = model_args['out_channels']
    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.logit = nn.Linear(feature_nums, args.num_classes)
    if self.do_ml:
      self.ml_logit = ArcMarginProduct(feature_nums, self.args.ml_num_classes)

  def gap_forward(self, x):
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    p_logits = self.logit(x)
    data = {'p_logits': p_logits}
    if self.do_ml:
      p_ml_logits = self.ml_logit(x)
      data['p_ml_logits'] = p_ml_logits
    return data

  def forward(self, data, **kwargs):
    outs = self.backbone.forward_features(data['image'])
    features = outs[-1]
    out = self.avgpool(features)  # B C 1
    x = torch.flatten(out, 1)

    logits = self.logit(x)

    data['logits'] = logits
    data['features'] = features
    if self.do_ml:
      ml_logits = self.ml_logit(x)
      data['ml_logits'] = ml_logits
    return data

def cls_swin_small_patch4_window7(args):
  model = SwinCls(args, feature_net='swin_small_patch4_window7')
  return model

def get_model(args, **kwargs):
  net = eval(args.model_name)(args)
  scheduler = None if args.scheduler is None else eval(args.scheduler)(net)
  loss = None if args.loss is None else eval(args.loss)()
  return net, scheduler, loss
