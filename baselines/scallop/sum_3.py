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

import scallopy

import csv
import time

mnist_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class MNISTSum3Dataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
  ):
    if train:
      length = 5000 * 3
    else:
      length = 500 * 3
    
    # Contains a MNIST dataset
    self.mnist_dataset = torch.utils.data.Subset(
      torchvision.datasets.MNIST(
        root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
      ),
      range(length)
    )
    self.index_map = list(range(len(self.mnist_dataset)))
    random.shuffle(self.index_map)

  def __len__(self):
    return int(len(self.mnist_dataset) / 3)

  def __getitem__(self, idx):
    # Get two data points
    (a_img, a_digit) = self.mnist_dataset[self.index_map[idx * 2]]
    (b_img, b_digit) = self.mnist_dataset[self.index_map[idx * 2 + 1]]
    (c_img, c_digit) = self.mnist_dataset[self.index_map[idx * 2 + 2]]

    # Each data has two images and the GT is the sum of two digits
    return (a_img, b_img, c_img, a_digit + b_digit + c_digit)

  @staticmethod
  def collate_fn(batch):
    a_imgs = torch.stack([item[0] for item in batch])
    b_imgs = torch.stack([item[1] for item in batch])
    c_imgs = torch.stack([item[2] for item in batch])
    digits = torch.stack([torch.tensor(item[3]).long() for item in batch])
    return ((a_imgs, b_imgs, c_imgs), digits)


def mnist_sum_3_loader(data_dir, batch_size_train, batch_size_test):
  train_loader = torch.utils.data.DataLoader(
    MNISTSum3Dataset(
      data_dir,
      train=True,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTSum3Dataset.collate_fn,
    batch_size=batch_size_train,
    shuffle=True
  )

  test_loader = torch.utils.data.DataLoader(
    MNISTSum3Dataset(
      data_dir,
      train=False,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTSum3Dataset.collate_fn,
    batch_size=batch_size_test,
    shuffle=True
  )

  return train_loader, test_loader


class MNISTNet(nn.Module):
  def __init__(self):
    super(MNISTNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
    self.fc1 = nn.Linear(1024, 1024)
    self.fc2 = nn.Linear(1024, 10)

  def forward(self, x):
    x = F.max_pool2d(self.conv1(x), 2)
    x = F.max_pool2d(self.conv2(x), 2)
    x = x.view(-1, 1024)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p = 0.5, training=self.training)
    x = self.fc2(x)
    return F.softmax(x, dim=1)


class MNISTSum3Net(nn.Module):
  def __init__(self, provenance, k):
    super(MNISTSum3Net, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()

    # Scallop Context
    self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    self.scl_ctx.add_relation("digit_1", int, input_mapping=[(i,) for i in range(10)])
    self.scl_ctx.add_relation("digit_2", int, input_mapping=[(i,) for i in range(10)])
    self.scl_ctx.add_relation("digit_3", int, input_mapping=[(i,) for i in range(10)])
    self.scl_ctx.add_rule("sum_3(a + b + c) = digit_1(a), digit_2(b), digit_3(c)")

    # The `sum_3` logical reasoning module
    self.sum_3 = self.scl_ctx.forward_function("sum_3", [(i,) for i in range(28)])

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    (a_imgs, b_imgs, c_imgs) = x

    # First recognize the two digits
    a_distrs = self.mnist_net(a_imgs)
    b_distrs = self.mnist_net(b_imgs)
    c_distrs = self.mnist_net(c_imgs)

    # Then execute the reasoning module; the result is a size 28 tensor
    return self.sum_3(digit_1=a_distrs, digit_2=b_distrs, digit_3=c_distrs)


def bce_loss(output, ground_truth):
  (_, dim) = output.shape
  gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
  return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
  return F.nll_loss(output, ground_truth)


class Trainer():
  def __init__(self, train_loader, test_loader, learning_rate, loss, k, provenance):
    self.network = MNISTSum3Net(provenance, k)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    if loss == "nll":
      self.loss = nll_loss
    elif loss == "bce":
      self.loss = bce_loss
    else:
      raise Exception(f"Unknown loss function `{loss}`")

  def train_epoch(self, epoch):
    self.network.train()
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    train_loss = 0
    for (data, target) in iter:
      self.optimizer.zero_grad()
      output = self.network(data)
      loss = self.loss(output, target)
      train_loss += loss.item()
      loss.backward()
      self.optimizer.step()
      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")
    return train_loss

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, target) in iter:
        output = self.network(data)
        test_loss += self.loss(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = 100. * correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")
    return correct.item() / num_items

  def train(self, n_epochs):
    dict = {}
    for epoch in range(1, n_epochs + 1):
      t0 = time.time()
      train_loss = self.train_epoch(epoch)
      t1 = time.time()
      dict['L ' + str(epoch)] = round(train_loss, ndigits=4)
      dict['T ' + str(epoch)] = round(t1 - t0, ndigits=4)
      acc = self.test_epoch(epoch)
      dict['A ' + str(epoch)] = round(acc, ndigits=6)
    return dict


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sum_3")
  parser.add_argument("--n-epochs", type=int, default=10)
  parser.add_argument("--batch-size-train", type=int, default=64)
  parser.add_argument("--batch-size-test", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=0.001)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--top-k", type=int, default=3)
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size_train = args.batch_size_train
  batch_size_test = args.batch_size_test
  learning_rate = args.learning_rate
  loss_fn = args.loss_fn
  k = args.top_k
  provenance = args.provenance
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))

  losses = ['L ' + str(i+1) for i in range(args.n_epochs)]
  accuracies = ['A ' + str(i+1) for i in range(args.n_epochs)]
  times = ['T ' + str(i+1) for i in range(args.n_epochs)]
  field_names = ['random seed'] + losses + accuracies + times

  dir_path = os.path.dirname(os.path.realpath(__file__))
  results_file =  dir_path + '/experiments10/sum_3.csv'

  # Dataloaders
  train_loader, test_loader = mnist_sum_3_loader(data_dir, batch_size_train, batch_size_test)

  # Create trainer and train
  trainer = Trainer(train_loader, test_loader, learning_rate, loss_fn, k, provenance)
  dict = trainer.train(n_epochs)
  dict['random seed'] = args.seed
  with open(results_file, 'a', newline='') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=field_names)
      writer.writerow(dict)
      csvfile.close()
