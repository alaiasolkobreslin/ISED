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
import csv

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

def add_mod_3(a, b):
  return (a + b) % 3

class MNISTAddMod3Dataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    length: int,
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
    self.length = length
    self.index_map = list(range(self.length * 2))
    random.shuffle(self.index_map)

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    # Get two data points
    (a_img, a_digit) = self.mnist_dataset[self.index_map[idx * 2]]
    (b_img, b_digit) = self.mnist_dataset[self.index_map[idx * 2 + 1]]

    return (a_img, b_img, (a_digit + b_digit) % 3)

  @staticmethod
  def collate_fn(batch):
    a_imgs = torch.stack([item[0] for item in batch])
    b_imgs = torch.stack([item[1] for item in batch])
    digits = torch.stack([torch.tensor(item[2]).long() for item in batch])
    return ((a_imgs, b_imgs), digits)


def mnist_add_mod_3_loader(data_dir, batch_size_train, batch_size_test):
    train_dataset = MNISTAddMod3Dataset(data_dir, length=5000, train=True, download=True, transform=mnist_img_transform)
    train_set_size = len(train_dataset)
    train_indices = list(range(train_set_size))
    split = int(train_set_size * 0.8)
    train_indices, val_indices = train_indices[:split], train_indices[split:]
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, train_indices), collate_fn=MNISTAddMod3Dataset.collate_fn, batch_size=batch_size_train, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, val_indices), collate_fn=MNISTAddMod3Dataset.collate_fn, batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        MNISTAddMod3Dataset(
        data_dir,
        length=500,
        train=False,
        download=True,
        transform=mnist_img_transform,
        ),
        collate_fn=MNISTAddMod3Dataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=True
    )

    return train_loader, valid_loader, test_loader


class MNISTAddMod3Net(nn.Module):
  def __init__(self):
    super(MNISTAddMod3Net, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = mnist_net.MNISTNet()


  def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
    (a_imgs, b_imgs) = x

    # First recognize the two digits
    a_distrs = self.mnist_net(a_imgs)
    b_distrs = self.mnist_net(b_imgs)

    return (a_distrs, b_distrs)


class RLAddMod3Net(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    self.saved_log_probs = []
    self.rewards = []
    self.perception = MNISTAddMod3Net()

  def forward(self, x):
    return self.perception.forward(x)


def validation(a, b):
    a = a.argmax(dim=1)
    b = b.argmax(dim=1)

    predictions = torch.stack([torch.tensor(add_mod_3(a[i], b[i])) for i in range(len(a))])
    return predictions

def final_output(model,ground_truth, args, a, b):
  d_a = torch.distributions.categorical.Categorical(a)
  d_b = torch.distributions.categorical.Categorical(b)

  s_a = d_a.sample()
  s_b = d_b.sample()

  model.saved_log_probs = d_a.log_prob(s_a)+d_b.log_prob(s_b)

  predictions = []
  for i in range(len(s_a)):
    prediction = add_mod_3(s_a[i], s_b[i])
    predictions.append(prediction)
    reward = common.compute_reward(prediction,ground_truth[i])
    model.rewards.append(reward)
  
  return torch.stack(predictions)


if __name__ == "__main__":
  parser = ArgumentParser('add_mod_3')
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
  
  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/mnist_add_mod_3"))
  os.makedirs(model_dir, exist_ok=True)

  torch.manual_seed(args.seed)
  random.seed(args.seed)

  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../benchmarks/data"))
  model_dir = os.path.join('model', 'nasr')
  outputs_dir = os.path.join('outputs', 'nasr')
  os.makedirs(model_dir, exist_ok=True)
  os.makedirs(outputs_dir, exist_ok=True)

  model = RLAddMod3Net()
  model.to(args.gpu_id)

  (train_loader, valid_loader, test_loader) = mnist_add_mod_3_loader(data_dir, args.batch_size, args.batch_size)
  trainer = common.Trainer(train_loader, valid_loader, test_loader, model, model_dir, final_output, args)
  results_dict = trainer.train(args.epochs)
  results_dict['task name'] = 'add_mod_3'
  results_dict['random seed'] = args.seed
  
  dir_path = os.path.dirname(os.path.realpath(__file__))
  results_file =  dir_path + '/experiments10/add_mod_3.csv'
  
  losses = ['L ' + str(i+1) for i in range(args.epochs)]
  accuracies = ['A ' + str(i+1) for i in range(args.epochs)]
  rewards = ['R ' + str(i+1) for i in range(args.epochs)]
  times = ['T ' + str(i+1) for i in range(args.epochs)]
  field_names = ['task name', 'random seed'] + losses + rewards + accuracies + times

  with open(results_file, 'w', newline='') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=field_names)
      writer.writeheader()
      writer.writerow(results_dict)
      csvfile.close()
  
