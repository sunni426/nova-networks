import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

def process_dataset(data, args):
  data['image'] = data['image'].transpose(2, 0, 1)  # HxWxC to CxHxW
  data['image'] = data['image'] / 255.
  data['image'] = torch.tensor(data['image']).float()

  if 'mask' in data and data['mask'] is not None:
    data['mask'] = torch.tensor(data['mask']).long()
  if 'labels' in data and data['labels'] is not None:
    data['labels'] = torch.tensor(data['labels'])
  if 'ml_labels' in data and data['ml_labels'] is not None:
    data['ml_labels'] = torch.tensor(data['ml_labels'])
  return data

class DistributedSequentialSampler(DistributedSampler):
  def __iter__(self):
    # deterministically shuffle based on epoch
    g = torch.Generator()
    g.manual_seed(self.epoch)
    if self.shuffle:
      indices = torch.randperm(len(self.dataset), generator=g).tolist()
    else:
      indices = list(range(len(self.dataset)))

    # add extra samples to make it evenly divisible
    indices += indices[:(self.total_size - len(indices))]
    assert len(indices) == self.total_size

    # subsample
    # indices = indices[self.rank:self.total_size:self.num_replicas]
    indices = indices[self.rank*self.num_samples:(self.rank+1)*self.num_samples]
    assert len(indices) == self.num_samples

    return iter(indices)
