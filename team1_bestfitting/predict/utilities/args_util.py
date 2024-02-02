from config.config import *

def set_default_attr(args):
  def _set_default_attr(args, key, value):
    if not args.__contains__(key):
      setattr(args, key, value)
    return args
  _set_default_attr(args, 'split_type', 'random')
  _set_default_attr(args, 'folds_num', 5)
  _set_default_attr(args, 'fold', 0)
  _set_default_attr(args, 'image_size', [512, 512])
  _set_default_attr(args, 'debug', False)
  _set_default_attr(args, 'in_channels', 4)
  _set_default_attr(args, 'num_classes', NUM_CLASSES)
  _set_default_attr(args, 'suffix', 'jpg')
  _set_default_attr(args, 'can_print', False)
  _set_default_attr(args, 'cell_version', None)
  _set_default_attr(args, 'split_df', None)
  _set_default_attr(args, 'specific_augment', 0)
  _set_default_attr(args, 'specific_augment_label', None)
  _set_default_attr(args, 'specific_augment_prob', 0)
  _set_default_attr(args, 'whole_train', 0)
  _set_default_attr(args, 'cell_complete', 0)
  return args
