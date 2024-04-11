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
from PIL import Image

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

mnist_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

def sum_2(a, b):
  return a + b

class MNISTSum2Dataset(torch.utils.data.Dataset):
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
    return int(len(self.mnist_dataset) / 2)

  def __getitem__(self, idx):
    # Get two data points
    (a_img, a_digit) = self.mnist_dataset[self.index_map[idx * 2]]
    (b_img, b_digit) = self.mnist_dataset[self.index_map[idx * 2 + 1]]

    # Each data has two images and the GT is the sum of two digits
    return (a_img, b_img, a_digit + b_digit)

  @staticmethod
  def collate_fn(batch):
    a_imgs = torch.stack([item[0] for item in batch])
    b_imgs = torch.stack([item[1] for item in batch])
    digits = torch.stack([torch.tensor(item[2]).long() for item in batch])
    return ((a_imgs, b_imgs), digits)


def mnist_sum_2_loader(data_dir, batch_size_train, batch_size_test):
  train_loader = torch.utils.data.DataLoader(
    MNISTSum2Dataset(
      data_dir,
      train=True,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTSum2Dataset.collate_fn,
    batch_size=batch_size_train,
    shuffle=True
  )

  test_loader = torch.utils.data.DataLoader(
    MNISTSum2Dataset(
      data_dir,
      train=False,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTSum2Dataset.collate_fn,
    batch_size=batch_size_test,
    shuffle=True
  )

  return train_loader, test_loader


class MNISTSum2Net(nn.Module):
  def __init__(self):
    super(MNISTSum2Net, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = mnist_net.MNISTNet()


  def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
    (a_imgs, b_imgs) = x

    # First recognize the two digits
    a_distrs = self.mnist_net(a_imgs)
    b_distrs = self.mnist_net(b_imgs)

    return (a_distrs, b_distrs)


class RLSum2Net(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    self.saved_log_probs = []
    self.rewards = []
    self.perception = MNISTSum2Net()

  def forward(self, x):
    return self.perception.forward(x)

def compute_reward(prediction, ground_truth):
    if prediction == ground_truth:
        reward = 1
    else:
        reward = 0
    return reward

def validation(a, b):
    a = a.argmax(dim=1)
    b = b.argmax(dim=1)

    predictions = torch.stack([torch.tensor(sum_2(a[i], b[i])) for i in range(len(a))])
    return predictions

def final_output(model,ground_truth, a, b, args):
  d_a = torch.distributions.categorical.Categorical(a)
  d_b = torch.distributions.categorical.Categorical(b)

  s_a = d_a.sample()
  s_b = d_b.sample()

  model.saved_log_probs = d_a.log_prob(s_a)+d_b.log_prob(s_b)

  predictions = []
  for i in range(len(s_a)):
    prediction = sum_2(s_a[i], s_b[i])
    predictions.append(torch.tensor(prediction))
    reward = compute_reward(prediction,ground_truth[i])
    model.rewards.append(reward)
  
  return torch.stack(predictions)

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.learning_rate
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    elif not args.disable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
    
class Trainer():
  def __init__(self, train_loader, test_loader, model, path, args):
    self.network = model
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.path = path
    self.args = args
    self.best_loss = None
    self.best_reward = None
    self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    self.criterion = nn.BCEWithLogitsLoss()
  
  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    train_loss = 0
    
    iter = tqdm(self.train_loader, total=len(self.train_loader))

    eps = np.finfo(np.float32).eps.item()
    for i, ((a_img, b_img), target) in enumerate(iter):
      images = (a_img.to(self.args.gpu_id), b_img.to(self.args.gpu_id))
      target = target.to(self.args.gpu_id)
      a, b = self.network(images)
      final_output(model,target,a,b,args)
      rewards = np.array(model.rewards)
      rewards_mean = rewards.mean()
      rewards = (rewards - rewards.mean())/(rewards.std() + eps)
      policy_loss = torch.zeros(len(rewards), requires_grad=True)
      
      for n, (reward, log_prob) in enumerate(zip(rewards, model.saved_log_probs)):
        policy_loss[n].data += (-log_prob.cpu()*reward)
      self.optimizer.zero_grad()
      
      policy_loss = policy_loss.sum()

      #if args.clip_grad_norm > 0:
      #  nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=args.clip_grad_norm, norm_type=2)

      num_items += a_img.size(0)
      train_loss += float(policy_loss.item() * a_img.size(0))
      policy_loss.backward()

      self.optimizer.step()
      
      avg_loss = train_loss/num_items
      iter.set_description(f"[Train Epoch {epoch}] AvgLoss: {avg_loss:.4f} AvgRewards: {rewards_mean:.4f}")
      
      if args.print_freq >= 0 and i % args.print_freq == 0:
        stats2 = {'epoch': epoch, 'train': i, 'avr_train_loss': avg_loss, 'avr_train_reward': rewards_mean}
        with open(f"model/nasr/detail_log.txt", "a") as f:
          f.write(json.dumps(stats2) + "\n")
      model.rewards = []
      model.shared_log_probs = []
      torch.cuda.empty_cache()
    
    return (train_loss/num_items), rewards_mean

  def test_epoch(self, epoch, time_begin):
    self.network.eval()
    num_items = 0
    test_loss = 0
    rewards_value = 0
    num_correct = 0

    eps = np.finfo(np.float32).eps.item()
    with torch.no_grad():
      for i, ((a_img, b_img), target) in enumerate(self.test_loader):
        images = (a_img.to(self.args.gpu_id), b_img.to(self.args.gpu_id))
        target = target.to(self.args.gpu_id)
        
        a, b = self.network(images)
        output = final_output(model,target,a,b,args)

        rewards = np.array(model.rewards)
        rewards_mean = rewards.mean()
        rewards_value += float(rewards_mean * a_img.size(0))
        rewards = (rewards - rewards.mean())/(rewards.std() + eps)

        policy_loss = []
        for reward, log_prob in zip(rewards, model.saved_log_probs):
          policy_loss.append(-log_prob*reward)
        policy_loss = (torch.stack(policy_loss)).sum()

        num_items += a_img.size(0)
        test_loss += float(policy_loss.item() * a_img.size(0))
        model.rewards = []
        model.saved_log_probs = []
        torch.cuda.empty_cache()

        # output = validation(f1, f2, f3)
        num_correct += (output==target).sum()
        perc = 100.*num_correct/num_items
        
        if self.best_loss is None or test_loss < self.best_loss:
          self.best_loss = test_loss
          torch.save(self.network.state_dict(), f'{self.path}/checkpoint_best_L.pth')
    
    avg_loss = (test_loss / num_items)
    avg_reward = (rewards_value/num_items)  
    total_mins = (time() - time_begin) / 60
    print(f"[Test Epoch {epoch}] {int(num_correct)}/{int(num_items)} ({perc:.2f})%")
    print(f'----[rl][Epoch {epoch}] \t \t AvgLoss {avg_loss:.4f} \t \t AvgReward {avg_reward:.4f} \t \t Time: {total_mins:.2f} ')
    
    return avg_loss, rewards_mean

  def train(self, n_epochs):
    time_begin = time()
    with open(f"{self.path}/log.txt", 'w'): pass
    with open(f"{self.path}/detail_log.txt", 'w'): pass
    for epoch in range(1, n_epochs+1):
      lr = adjust_learning_rate(self.optimizer, epoch, self.args)
      train_loss = self.train_epoch(epoch)
      test_loss = self.test_epoch(epoch, time_begin)
      stats = {'epoch': epoch, 'lr': lr, 'train_loss': train_loss, 'val_loss': test_loss, 'best_loss': self.best_loss}
      with open(f"{self.path}/log.txt", "a") as f: 
        f.write(json.dumps(stats) + "\n")
    torch.save(self.network.state_dict(), f'{self.path}/checkpoint_last.pth')

if __name__ == "__main__":
  parser = ArgumentParser('leaf')
  parser.add_argument('--gpu-id', default='cuda:0', type=str)
  parser.add_argument('-j', '--workers', default=0, type=int)
  parser.add_argument('--print-freq', default=5, type=int)

  parser.add_argument('--epochs', default=10, type=int)
  parser.add_argument('--warmup', default=10, type=int)
  parser.add_argument('-b', '--batch-size', default=16, type=int)
  parser.add_argument('--learning-rate', default=0.0001, type=float)
  parser.add_argument('--weight-decay', default=3e-1, type=float)
  parser.add_argument('--clip-grad-norm', default=1., type=float)
  parser.add_argument('--disable-cos', action='store_true')

  args = parser.parse_args()
  
  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/mnist_sum_2"))
  os.makedirs(model_dir, exist_ok=True)

  torch.manual_seed(1234)
  random.seed(1234)

  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../benchmarks/data"))
  model_dir = os.path.join('demo/model', 'nasr')
  os.makedirs(model_dir, exist_ok=True)

  model = RLSum2Net()
  model.to(args.gpu_id)

  (train_loader, test_loader) = mnist_sum_2_loader(data_dir, args.batch_size, args.batch_size)
  trainer = Trainer(train_loader, test_loader, model, model_dir, args)
  trainer.train(args.epochs)