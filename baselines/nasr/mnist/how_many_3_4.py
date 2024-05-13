def how_many_3_or_4(x):
    return sum((n == 3 or n == 4) for n in x)


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

def how_many_3_4(a, b, c, d, e, f, g, h):
  digit_lst = [a, b, c, d, e, f, g, h]
  return sum((n == 3 or n == 4) for n in digit_lst)

class MNISTHowMany3Or4Dataset(torch.utils.data.Dataset):
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
    self.index_map = list(range(self.length * 8))
    random.shuffle(self.index_map)

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    (a_img, a_digit) = self.mnist_dataset[self.index_map[idx * 8]]
    (b_img, b_digit) = self.mnist_dataset[self.index_map[idx * 8 + 1]]
    (c_img, c_digit) = self.mnist_dataset[self.index_map[idx * 8 + 2]]
    (d_img, d_digit) = self.mnist_dataset[self.index_map[idx * 8 + 3]]
    (e_img, e_digit) = self.mnist_dataset[self.index_map[idx * 8 + 4]]
    (f_img, f_digit) = self.mnist_dataset[self.index_map[idx * 8 + 5]]
    (g_img, g_digit) = self.mnist_dataset[self.index_map[idx * 8 + 6]]
    (h_img, h_digit) = self.mnist_dataset[self.index_map[idx * 8 + 7]]

    digit_lst = [a_digit, b_digit, c_digit, d_digit, e_digit, f_digit, g_digit, h_digit]
    result = sum((n == 3 or n == 4) for n in digit_lst)

    return (a_img, b_img, c_img, d_img, e_img, f_img, g_img, h_img, result)

  @staticmethod
  def collate_fn(batch):
    a_imgs = torch.stack([item[0] for item in batch])
    b_imgs = torch.stack([item[1] for item in batch])
    c_imgs = torch.stack([item[2] for item in batch])
    d_imgs = torch.stack([item[3] for item in batch])
    e_imgs = torch.stack([item[4] for item in batch])
    f_imgs = torch.stack([item[5] for item in batch])
    g_imgs = torch.stack([item[6] for item in batch])
    h_imgs = torch.stack([item[7] for item in batch])
    digits = torch.stack([torch.tensor(item[8]).long() for item in batch])
    return ((a_imgs, b_imgs, c_imgs, d_imgs, e_imgs, f_imgs, g_imgs, h_imgs), digits)


def mnist_how_many_3_4_loader(data_dir, batch_size_train, batch_size_test):
    train_dataset = MNISTHowMany3Or4Dataset(data_dir, length=5000, train=True, download=True, transform=mnist_img_transform)
    train_set_size = len(train_dataset)
    train_indices = list(range(train_set_size))
    split = int(train_set_size * 0.8)
    train_indices, val_indices = train_indices[:split], train_indices[split:]
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, train_indices), collate_fn=MNISTHowMany3Or4Dataset.collate_fn, batch_size=batch_size_train, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, val_indices), collate_fn=MNISTHowMany3Or4Dataset.collate_fn, batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        MNISTHowMany3Or4Dataset(
        data_dir,
        length=500,
        train=False,
        download=True,
        transform=mnist_img_transform,
        ),
        collate_fn=MNISTHowMany3Or4Dataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=True
    )

    return train_loader, valid_loader, test_loader


class MNISTHowMany3Or4Net(nn.Module):
  def __init__(self):
    super(MNISTHowMany3Or4Net, self).__init__()

    self.mnist_net = mnist_net.MNISTNet()

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
    (a_imgs, b_imgs, c_imgs, d_imgs, e_imgs, f_imgs, g_imgs, h_imgs) = x

    a_distrs = self.mnist_net(a_imgs)
    b_distrs = self.mnist_net(b_imgs)
    c_distrs = self.mnist_net(c_imgs)
    d_distrs = self.mnist_net(d_imgs)
    e_distrs = self.mnist_net(e_imgs)
    f_distrs = self.mnist_net(f_imgs)
    g_distrs = self.mnist_net(g_imgs)
    h_distrs = self.mnist_net(h_imgs)

    return (a_distrs, b_distrs, c_distrs, d_distrs, e_distrs, f_distrs, g_distrs, h_distrs)


class RLHowMany3Or4Net(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    self.saved_log_probs = []
    self.rewards = []
    self.perception = MNISTHowMany3Or4Net()

  def forward(self, x):
    return self.perception.forward(x)


def validation(a, b, c, d, e, f, g, h):
    a = a.argmax(dim=1)
    b = b.argmax(dim=1)
    c = c.argmax(dim=1)
    d = d.argmax(dim=1)
    e = e.argmax(dim=1)
    f = f.argmax(dim=1)
    g = g.argmax(dim=1)
    h = h.argmax(dim=1)

    predictions = torch.stack([torch.tensor(how_many_3_4(a[i], b[i], c[i], d[i], e[i], f[i], g[i], h[i])) for i in range(len(a))])
    return predictions

def final_output(model,ground_truth, args, a, b, c, d, e, f, g, h):
  d_a = torch.distributions.categorical.Categorical(a)
  d_b = torch.distributions.categorical.Categorical(b)
  d_c = torch.distributions.categorical.Categorical(c)
  d_d = torch.distributions.categorical.Categorical(d)
  d_e = torch.distributions.categorical.Categorical(e)
  d_f = torch.distributions.categorical.Categorical(f)
  d_g = torch.distributions.categorical.Categorical(g)
  d_h = torch.distributions.categorical.Categorical(h)

  s_a = d_a.sample()
  s_b = d_b.sample()
  s_c = d_c.sample()
  s_d = d_d.sample()
  s_e = d_e.sample()
  s_f = d_f.sample()
  s_g = d_g.sample()
  s_h = d_h.sample()

  model.saved_log_probs = d_a.log_prob(s_a)+d_b.log_prob(s_b)+d_c.log_prob(s_c)+d_d.log_prob(s_d)+d_e.log_prob(s_e)+d_f.log_prob(s_f)+d_g.log_prob(s_g)+d_h.log_prob(s_h)

  predictions = []
  for i in range(len(s_a)):
    prediction = how_many_3_4(s_a[i], s_b[i], s_c[i], s_d[i], s_e[i], s_f[i], s_g[i], s_h[i])
    predictions.append(prediction)
    reward = common.compute_reward(prediction,ground_truth[i])
    model.rewards.append(reward)
  
  return torch.stack(predictions)


if __name__ == "__main__":
  parser = ArgumentParser('how_many_3_4')
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
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/mnist_how_many_3_4"))
  os.makedirs(model_dir, exist_ok=True)

  torch.manual_seed(args.seed)
  random.seed(args.seed)

  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../benchmarks/data"))
  model_dir = os.path.join('model', 'nasr')
  outputs_dir = os.path.join('outputs', 'nasr')
  os.makedirs(model_dir, exist_ok=True)
  os.makedirs(outputs_dir, exist_ok=True)

  model = RLHowMany3Or4Net()
  model.to(args.gpu_id)

  (train_loader, valid_loader, test_loader) = mnist_how_many_3_4_loader(data_dir, args.batch_size, args.batch_size)
  trainer = common.Trainer(train_loader, valid_loader, test_loader, model, model_dir, final_output, args)
  results_dict = trainer.train(args.epochs)
  results_dict['task name'] = 'how_many_3_4'
  results_dict['random seed'] = args.seed
  
  dir_path = os.path.dirname(os.path.realpath(__file__))
  results_file =  dir_path + '/experiments10/how_many_3_4.csv'
  
  losses = ['L ' + str(i+1) for i in range(args.epochs)]
  accuracies = ['A ' + str(i+1) for i in range(args.epochs)]
  rewards = ['R ' + str(i+1) for i in range(args.epochs)]
  times = ['T ' + str(i+1) for i in range(args.epochs)]
  field_names = ['task name', 'random seed'] + losses + rewards + accuracies + times

  with open(results_file, 'a', newline='') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=field_names)
      writer.writerow(results_dict)
      csvfile.close()
  