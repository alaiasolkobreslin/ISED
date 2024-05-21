import os
import random
from typing import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
import time

import blackbox

mnist_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class MNISTDigitsDataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    digit: int,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False
  ):
    # Contains a MNIST dataset
    if train: self.length = min(5000 * digit, 60000)
    else: self.length = min(500 * digit, 10000)
    self.digit = digit
    self.mnist_dataset = torch.utils.data.Subset(
      torchvision.datasets.MNIST(
        root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
      ),
      range(self.length)
    )
    self.index_map = list(range(len(self.mnist_dataset)))
    random.shuffle(self.index_map)

    self.sum_dataset = []
    for i in range(len(self.mnist_dataset)//self.digit):
      self.sum_dataset.append([])
      for j in range(self.digit):
        self.sum_dataset[i].append(self.mnist_dataset[self.index_map[i*self.digit + j]])

  def __len__(self):
    return len(self.sum_dataset)

  def __getitem__(self, idx):
    # Get two data points
    item = self.sum_dataset[idx]
    data, target = [], []
    for (d,t) in item:
      data.append(d)
      target.append(t)
    
    target = sum_n(*tuple(target))

    # Each data has two images and the GT is the sum of two digits
    return (*tuple(data), target)

  @staticmethod
  def collate_fn(batch):
    imgs = []
    for i in range(len(batch[0])-1):
      imgs.append(torch.stack([item[i] for item in batch]))
    digits = torch.stack([torch.tensor(item[-1]).long() for item in batch])
    return (tuple(imgs), digits)

def mnist_digits_loader(data_dir, batch_size, digit):
  train_loader = torch.utils.data.DataLoader(
    MNISTDigitsDataset(
      data_dir,
      digit,
      train=True,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTDigitsDataset.collate_fn,
    batch_size=batch_size,
    shuffle=True
  )

  test_loader = torch.utils.data.DataLoader(
    MNISTDigitsDataset(
      data_dir,
      digit,
      train=False,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTDigitsDataset.collate_fn,
    batch_size=batch_size,
    shuffle=True
  )

  return train_loader, test_loader

class MNISTNet(nn.Module):
  def __init__(self):
    super(MNISTNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
    self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
    self.fc1 = nn.Linear(256, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, 256)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return F.softmax(x, dim=1)

def sum_n(*nums):
  return sum(nums)

class MNISTSum2Net(nn.Module):
  def __init__(self, loss_aggregator, sample_count, digit):
    super(MNISTSum2Net, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()
    self.digit = digit
    self.sample_count = sample_count
    self.input_mapping = [blackbox.DiscreteInputMapping(list(range(10)))]*self.digit
    self.loss_aggregator = loss_aggregator

    # Scallop Context
    self.sum_2 = blackbox.BlackBoxFunction(
      sum_n,
      tuple(self.input_mapping),
      blackbox.DiscreteOutputMapping(list(range(self.digit*9+1))),
      sample_count=self.sample_count,
      loss_aggregator=self.loss_aggregator)

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
    batch_size = x[0].shape[0]
    x = torch.cat(x, dim=0)

    # First recognize the two digits
    x = self.mnist_net(x)
    x = [x[i*batch_size:(i + 1) * batch_size,:] for i in range(self.digit)]

    # Then execute the reasoning module; the result is a size 19 tensor
    return self.sum_2(*tuple(x)) # Tensor 64 x 19

class Trainer():
  def __init__(self, train_loader, test_loader, model_dir, learning_rate, loss_aggregator, sample_count, digit, seed):
    self.model_dir = model_dir
    self.device = torch.device("cpu")
    self.network = MNISTSum2Net(loss_aggregator, sample_count, digit)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = 10000000000
    self.seed = seed

  def loss(self, output, ground_truth):
    (_, dim) = output.shape
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
    return F.binary_cross_entropy(output, gt)

  def train_epoch(self, epoch):
    self.network.train()
    train_loss = 0
    print(f"[Train Epoch {epoch}]")
    for (data, target) in train_loader:
      self.optimizer.zero_grad()
      output = self.network(data)
      loss = self.loss(output, target)
      train_loss += loss
      loss.backward()
      self.optimizer.step()
    return train_loss

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for (data, target) in test_loader:
        output = self.network(data)
        test_loss += self.loss(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = 100. * correct / num_items
      print(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")

    if self.best_loss is None or self.best_loss < perc:
      self.best_loss = perc
      torch.save(self.network, model_dir+f"/{self.seed}_best.pkl")
    
    return float(correct / num_items)

  def train(self, n_epochs):
    for epoch in range(1, n_epochs + 1):
      t0 = time.time()
      train_loss = self.train_epoch(epoch)
      t1 = time.time()
      acc = self.test_epoch(epoch)
    print(f"Test accuracy: {acc}")
    torch.save(self.network, model_dir+f"/{self.seed}_last.pkl")


if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("sum_m")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--dispatch", type=str, default="parallel")
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--digits", type=int, default=16)
    parser.add_argument("--agg", type=str, default="add_mult")
    args = parser.parse_args()

    # Parameters
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    loss_aggregator = args.agg
    sample_count = args.sample_count
    digits = args.digits
    seed = args.seed

    torch.manual_seed(seed)
    random.seed(seed)
        
    task_type = f'sum_{digits}'

    # Data
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
    model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/sum_m"))
    os.makedirs(model_dir, exist_ok=True)

    # Dataloaders
    train_loader, test_loader = mnist_digits_loader(data_dir, batch_size, digits)

    # Create trainer and train
    trainer = Trainer(train_loader, test_loader, model_dir, learning_rate, loss_aggregator, sample_count, digits, seed)
    trainer.train(n_epochs)