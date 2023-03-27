import os
import json
import random
from typing import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from tqdm import tqdm

from util import sample

# Problem: https://leetcode.com/problems/add-two-numbers/

def add_two_numbers_forward(inputs):
  size = int(len(inputs) / 2)
  n_samples = inputs[0].shape[0]
  
  a_inputs = torch.stack(inputs[:size])
  b_inputs = torch.stack(inputs[size:])

  a_nums = torch.zeros(n_samples)
  b_nums = torch.zeros(n_samples)

  for i in range(size):
    a_nums = a_nums + a_inputs[i] * (10 ** i)
    b_nums = b_nums + b_inputs[i] * (10 ** i)

  return a_nums + b_nums

mnist_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class MNISTAddTwoNumbersDataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    n_digits: int,
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
    self.n_digits = n_digits

  def __len__(self):
    return int(len(self.mnist_dataset) / (self.n_digits * 2))

  def __getitem__(self, idx):
    # Get two data points
    total_digits = self.n_digits * 2

    imgs_digits = [self.mnist_dataset[self.index_map[idx * total_digits + i]] for i in range(total_digits)]
    imgs = torch.stack([imgs_digits[i][0] for i in range(total_digits)])
    digits = [imgs_digits[i][1] for i in range(total_digits)]

    a_num = b_num = 0
    a_digits = digits[:self.n_digits]
    b_digits = digits[self.n_digits:]
    for i in range(self.n_digits):
      a_num = a_num + a_digits[i] * (10 ** i)
      b_num = b_num + b_digits[i] * (10 ** i)

    result = a_num + b_num
    return (imgs, result)

  @staticmethod
  def collate_fn(batch):
    total_digits = batch[0][0].shape[0]
    imgs = torch.stack([torch.stack([item[0][i] for item in batch]) for i in range(total_digits)])
    digits = [item[1] for item in batch]
    return (imgs, digits)


def MNIST_add_two_numbers_loader(data_dir, n_digits, batch_size_train, batch_size_test):
  train_loader = torch.utils.data.DataLoader(
    MNISTAddTwoNumbersDataset(
      data_dir,
      n_digits,
      train=True,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTAddTwoNumbersDataset.collate_fn,
    batch_size=batch_size_train,
    shuffle=True
  )

  test_loader = torch.utils.data.DataLoader(
    MNISTAddTwoNumbersDataset(
      data_dir,
      n_digits,
      train=False,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTAddTwoNumbersDataset.collate_fn,
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


class MNISTAddTwoNumbersNet(nn.Module):
  def __init__(self, n_digits):
    super(MNISTAddTwoNumbersNet, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()

    input_mapping = [10] * (n_digits * 2)

    self.sampling = sample.Sample(n_inputs=n_digits*2, n_samples=args.n_samples, input_mapping=input_mapping, fn=add_two_numbers_forward)

  def add_two_numbers_test(self, digits):
    return self.sampling.sample_test(digits)

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor], y: List[int]):
    """
    Invoked during training

    Takes in input pair of images (x_a, x_b) and the ground truth output sum (r)
    Returns the loss (a single scalar)
    """
    imgs = x
    total_digits = n_digits * 2

    # First recognize the digits
    batched = torch.cat(tuple(imgs))
    distrs = self.mnist_net(batched).split(imgs[0].shape[0])
    distrs_list = [distr.clone().detach() for distr in distrs]

    argss = list(zip(*(tuple(distrs_list)), y))
    out_pred = map(self.sampling.sample_train, argss)
    out_pred = list(zip(*out_pred))
    preds = [torch.stack(out_pred[i]).view([distrs[i].shape[0], 10]) for i in range(total_digits)]

    cat_distrs = torch.cat(distrs)
    cat_pred = torch.cat(preds)
    l = F.mse_loss(cat_distrs,cat_pred)
    return l

  def evaluate(self, x: Tuple[torch.Tensor, torch.Tensor]):
    """
    Invoked during testing

    Takes in input pair of images (x_a, x_b)
    Returns the predicted sum of the two digits (vector of dimension 19)
    """
    imgs = x
    total_digits = n_digits * 2

    # First recognize the two digits
    distrs = [self.mnist_net(imgs[i]) for i in range(total_digits)]

    # Testing: execute the reasoning module; the result is a size 19 tensor
    return self.add_two_numbers_test(distrs) # Tensor 64 x 19


class Trainer():
  def __init__(self, train_loader, test_loader, learning_rate, n_digits):
    self.network = MNISTAddTwoNumbersNet(n_digits)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader

  def train_epoch(self, epoch):
    self.network.train()
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    total_loss = 0.0
    for (batch_id, (data, target)) in enumerate(iter):
      self.optimizer.zero_grad()
      loss = self.network.forward(data, target)
      loss.backward()
      for param in self.network.parameters():
        param.grad.data.clamp_(-1, 1)
      self.optimizer.step()
      total_loss += loss.item()
      avg_loss = total_loss / (batch_id + 1)
      iter.set_description(f"[Train Epoch {epoch}] Avg Loss: {avg_loss:.4f}, Batch Loss: {loss.item():.4f}")

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = 0
    correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, target) in iter:
        batch_size = len(target)
        output = self.network.evaluate(data) # Float Tensor 64 x 19
        for i in range(batch_size):
          if output[i].item() == target[i]:
            correct += 1
        num_items += batch_size
        perc = 100. * correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Accuracy: {correct}/{num_items} ({perc:.2f}%)")

  def train(self, n_epochs):
    # self.test_epoch(0)
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_add_two_numbers_sampling")
  parser.add_argument("--n-epochs", type=int, default=10)
  parser.add_argument("--batch-size-train", type=int, default=64)
  parser.add_argument("--batch-size-test", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--n-samples", type=int, default=100)
  parser.add_argument("--difficulty", type=str, default="easy")
  args = parser.parse_args()

  # Read json
  dir_path = os.path.dirname(os.path.realpath(__file__))
  data = json.load(open(os.path.join(dir_path ,os.path.join('specs', 'add_two_numbers.json'))))

  # Parameters
  n_epochs = args.n_epochs
  batch_size_train = args.batch_size_train
  batch_size_test = args.batch_size_test
  learning_rate = args.learning_rate
  n_digits = data['n_digits'][args.difficulty]
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))

  # Dataloaders
  train_loader, test_loader = MNIST_add_two_numbers_loader(data_dir, n_digits, batch_size_train, batch_size_test)

  # Create trainer and train
  trainer = Trainer(train_loader, test_loader, learning_rate, n_digits)
  trainer.train(n_epochs)
