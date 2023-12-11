import torch
from torch import nn
import itertools
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import blackbox

class PreNorm(nn.Module):
  def __init__(self, dim, fn):
    super().__init__()
    self.norm = nn.LayerNorm(dim)
    self.fn = fn

  def forward(self, x, **kwargs):
    return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
  def __init__(self, dim, hidden_dim, dropout = 0.):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(dim, hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, dim),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)
  
class Attention(nn.Module):
  def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
    super().__init__()
    inner_dim = dim_head *  heads
    project_out = not (heads == 1 and dim_head == dim)

    self.heads = heads
    self.scale = dim_head ** -0.5

    self.attend = nn.Softmax(dim = -1)
    self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

    self.to_out = nn.Sequential(
      nn.Linear(inner_dim, dim),
      nn.Dropout(dropout)
    ) if project_out else nn.Identity()

  def forward(self, x):
    qkv = self.to_qkv(x).chunk(3, dim = -1)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
    dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
    attn = self.attend(dots)
    out = torch.matmul(attn, v)
    out = rearrange(out, 'b h n d -> b n (h d)')
    return self.to_out(out)

class Transformer(nn.Module):
  def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
    super().__init__()
    self.layers = nn.ModuleList([])
    for _ in range(depth):
      self.layers.append(nn.ModuleList([
        PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
        PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
      ]))

  def forward(self, x):
    for attn, ff in self.layers:
      x = attn(x) + x
      x = ff(x) + x
    return x

def build_adj(num_block_x, num_block_y):
  adjacency = []
  block_coord_to_block_id = lambda x, y: y * num_block_x + x
  for i, j in itertools.product(range(num_block_x), range(num_block_y)):
    for (dx, dy) in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
      x, y = i + dx, j + dy
      if x >= 0 and x < num_block_x and y >= 0 and y < num_block_y:
        source_id = block_coord_to_block_id(i, j)
        target_id = block_coord_to_block_id(x, y)
        adjacency.append((source_id, target_id))
  return adjacency

def existsPath(is_connected, is_endpoint):
  return torch.randint(0,(64,)).long()

class PathFinderNet(nn.Module):
  def __init__(self, sample_count, num_block_x=6, num_block_y=6):
    super(PathFinderNet, self).__init__()

    # block
    self.num_block_x = num_block_x
    self.num_block_y = num_block_y
    self.num_blocks = num_block_x * num_block_y
    self.block_coord_to_block_id = lambda x, y: y * num_block_x + x

    # Adjacency
    self.adjacency = build_adj(num_block_x, num_block_y)

    # Scallop Context
    self.bbox_connected = blackbox.BlackBoxFunction(
      existsPath,
      (blackbox.DiscreteInputMapping(list(range(120))),blackbox.DiscreteInputMapping(list(range(36)))),
      blackbox.DiscreteOutputMapping(list(range(0))),
      sample_count=sample_count
    )
    

class CNNPathFinder32Net(PathFinderNet):
  def __init__(self, sample_count, num_block_x=6, num_block_y=6):
    super(CNNPathFinder32Net, self).__init__(sample_count, num_block_x, num_block_y)

    # CNN
    self.cnn = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=5),
      nn.Conv2d(32, 32, kernel_size=5),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, kernel_size=5),
      nn.Conv2d(64, 64, kernel_size=5),
      nn.MaxPool2d(2),
      nn.Flatten(),
    )

    # Fully connected for `is_endpoint`
    self.is_endpoint_fc = nn.Sequential(
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, self.num_blocks),
      nn.Sigmoid(),
    )

    # Fully connected for `connectivity`
    self.is_connected_fc = nn.Sequential(
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, len(self.adjacency)),
      nn.Sigmoid(),
    )

  def forward(self, image):
    embedding = self.cnn(image)
    is_connected = self.is_connected_fc(embedding) # 64 * 120
    is_endpoint = self.is_endpoint_fc(embedding) # 64 * 36
    result = self.bbox_connected(is_connected, is_endpoint)
    return result