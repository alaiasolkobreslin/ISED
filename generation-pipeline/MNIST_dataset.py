import random
import os
from typing import *

import torch
import torchvision

mnist_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class MNISTDataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
  ):
    # Contains a MNIST dataset
    self.mnist_dataset = torchvision.datasets.MNIST(
      root,
      train=train,
      transform=transform,
      target_transform=target_transform,
      download=download,
    )
    self.index_map = list(range(len(self.mnist_dataset)))
    random.shuffle(self.index_map)

  def __len__(self):
    return len(self.mnist_dataset)

  def __getitem__(self, idx):
    return self.mnist_dataset[self.index_map[idx]]
  
def get_data():
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../data"))
    data = MNISTDataset(
      data_dir,
      train=True,
      download=True,
      transform=mnist_img_transform,
    )
    sorted = torch.sort(data.mnist_dataset.targets)
    idxs = sorted.indices
    values = sorted.values
    ids_of_digit = [None] * 10
    for i in range(10):
      t = (values == i).nonzero(as_tuple=True)[0]
      ids_of_digit[i] = idxs[t[0]:t[-1]]
    return (data.mnist_dataset, ids_of_digit)
