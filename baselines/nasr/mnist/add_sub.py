import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision

import math
import os
import json
import numpy as np
from time import time
import random
from typing import Optional, Callable

from argparse import ArgumentParser

import os
import random
from typing import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from tqdm import tqdm

import mnist_net
import common

mnist_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

def add_sub(a, b, c):
  return a + b - c

class MNISTAddSubDataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    length: int,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
  ):
    self.mnist_dataset = torchvision.datasets.MNIST(
      root,
      train=train,
      transform=transform,
      target_transform=target_transform,
      download=download,
    )
    self.length = length
    self.index_map = list(range(self.length * 3))
    random.shuffle(self.index_map)

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    (a_img, a_digit) = self.mnist_dataset[self.index_map[idx * 3]]
    (b_img, b_digit) = self.mnist_dataset[self.index_map[idx * 3 + 1]]
    (c_img, c_digit) = self.mnist_dataset[self.index_map[idx * 3 + 2]]

    return (a_img, b_img, c_img, a_digit + b_digit - c_digit)

  @staticmethod
  def collate_fn(batch):
    a_imgs = torch.stack([item[0] for item in batch])
    b_imgs = torch.stack([item[1] for item in batch])
    c_imgs = torch.stack([item[2] for item in batch])
    digits = torch.stack([torch.tensor(item[3]).long() for item in batch])
    return ((a_imgs, b_imgs, c_imgs), digits)


def mnist_add_sub_loader(data_dir, batch_size_train, batch_size_test):
    train_dataset = MNISTAddSubDataset(data_dir, length=5000, train=True, download=True, transform=mnist_img_transform)
    train_set_size = len(train_dataset)
    train_indices = list(range(train_set_size))
    split = int(train_set_size * 0.8)
    train_indices, val_indices = train_indices[:split], train_indices[split:]
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, train_indices), collate_fn=MNISTAddSubDataset.collate_fn, batch_size=batch_size_train, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, val_indices), collate_fn=MNISTAddSubDataset.collate_fn, batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        MNISTAddSubDataset(
        data_dir,
        length=500,
        train=False,
        download=True,
        transform=mnist_img_transform,
        ),
        collate_fn=MNISTAddSubDataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=True
    )

    return train_loader, valid_loader, test_loader


class MNISTAddSubNet(nn.Module):
  def __init__(self):
    super(MNISTAddSubNet, self).__init__()

    self.mnist_net = mnist_net.MNISTNet()

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
    (a_imgs, b_imgs, c_imgs) = x

    a_distrs = self.mnist_net(a_imgs)
    b_distrs = self.mnist_net(b_imgs)
    c_distrs = self.mnist_net(c_imgs)

    return (a_distrs, b_distrs, c_distrs)


class RLAddSubNet(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    self.saved_log_probs = []
    self.rewards = []
    self.perception = MNISTAddSubNet()

  def forward(self, x):
    return self.perception.forward(x)


def validation(a, b, c):
    a = a.argmax(dim=1)
    b = b.argmax(dim=1)
    c = c.argmax(dim=1)

    predictions = torch.stack([torch.tensor(add_sub(a[i], b[i], c[i])) for i in range(len(a))])
    return predictions

def final_output(model,ground_truth, args, a, b, c):
  d_a = torch.distributions.categorical.Categorical(a)
  d_b = torch.distributions.categorical.Categorical(b)
  d_c = torch.distributions.categorical.Categorical(c)

  s_a = d_a.sample()
  s_b = d_b.sample()
  s_c = d_c.sample()

  model.saved_log_probs = d_a.log_prob(s_a)+d_b.log_prob(s_b)+d_c.log_prob(s_c)

  predictions = []
  for i in range(len(s_a)):
    prediction = add_sub(s_a[i], s_b[i], s_c[i])
    predictions.append(prediction)
    reward = common.compute_reward(prediction,ground_truth[i])
    model.rewards.append(reward)
  
  return torch.stack(predictions)


if __name__ == "__main__":
  parser = ArgumentParser('add_sub')
  parser.add_argument('--gpu-id', default='cuda:0', type=str)
  parser.add_argument('-j', '--workers', default=0, type=int)
  parser.add_argument('--print-freq', default=5, type=int)
  parser.add_argument('--seed', default=1234, type=int)

  parser.add_argument('--epochs', default=100, type=int)
  parser.add_argument('--warmup', default=0, type=int)
  parser.add_argument('-b', '--batch-size', default=16, type=int)
  parser.add_argument('--learning-rate', default=0.0001, type=float)
  parser.add_argument('--weight-decay', default=3e-1, type=float)
  parser.add_argument('--clip-grad-norm', default=1., type=float)
  parser.add_argument('--disable-cos', action='store_true')

  args = parser.parse_args()
  
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../../data"))
  model_dir = os.path.join('model', 'nasr')
  outputs_dir = os.path.join('outputs', 'nasr')
  os.makedirs(model_dir, exist_ok=True)
  os.makedirs(outputs_dir, exist_ok=True)

  model = RLAddSubNet()
  model.to(args.gpu_id)

  (train_loader, valid_loader, test_loader) = mnist_add_sub_loader(data_dir, args.batch_size, args.batch_size)
  trainer = common.Trainer(train_loader, valid_loader, test_loader, model, model_dir, final_output, args)
  trainer.train(args.epochs)
