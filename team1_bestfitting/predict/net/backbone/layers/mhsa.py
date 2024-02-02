import torch
import torch.nn as nn

# https://github.com/leaderj1001/BottleneckTransformers/blob/main/model.py
class MHSA(nn.Module):
  def __init__(self, n_dims, width=14, height=14, pos_enc_type='absolute'):
    super(MHSA, self).__init__()
    assert pos_enc_type in ['relative', 'absolute'], pos_enc_type
    self.pos_enc_type = pos_enc_type

    self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
    self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
    self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

    self.rel_h = nn.Parameter(torch.randn([1, n_dims, 1, height]), requires_grad=True)
    self.rel_w = nn.Parameter(torch.randn([1, n_dims, width, 1]), requires_grad=True)

    self.softmax = nn.Softmax(dim=-1)

  def absolute_content_position(self, q, C):
    content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)
    content_position = torch.matmul(content_position, q)
    return content_position

  def relative_content_position(self, q, C):
    raise NotImplementedError()

  def forward(self, x):
    n_batch, C, width, height = x.size()
    q = self.query(x).view(n_batch, C, -1)
    k = self.key(x).view(n_batch, C, -1)
    v = self.value(x).view(n_batch, C, -1)

    content_content = torch.bmm(q.permute(0, 2, 1), k)

    if self.pos_enc_type == 'absolute':
      content_position = self.absolute_content_position(q, C)
    elif self.pos_enc_type == 'relative':
      content_position = self.relative_content_position(q, C)
    else:
      raise ValueError(self.pos_enc_type)

    energy = content_content + content_position
    attention = self.softmax(energy)

    out = torch.bmm(v, attention.permute(0, 2, 1))
    out = out.view(n_batch, C, width, height)

    return out
