import os
import random
from typing import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.categorical import Categorical

from argparse import ArgumentParser
from tqdm import tqdm

import scallopy

mnist_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

def sample_fn(arg):
  a_dist, b_dist, c_dist, y = arg[0], arg[1], arg[2], arg[3]

  a_cat, b_cat, c_cat = Categorical(probs=a_dist), Categorical(probs=b_dist), Categorical(probs=c_dist)
  a_samples, b_samples, c_samples = a_cat.sample((args.n_samples,)), b_cat.sample((args.n_samples,)), c_cat.sample((args.n_samples,))
  a_on, b_on, c_on = dict.fromkeys(range(10), 0), dict.fromkeys(range(10), 0), dict.fromkeys(range(10), 0)
  for i in range(args.n_samples):
    if a_samples[i] + b_samples[i] + c_samples[i] == y:
      a_on[a_samples[i].item()] += 1
      b_on[b_samples[i].item()] += 1
      c_on[c_samples[i].item()] += 1

  total = sum(a_on.values())
  
  if total:
    a_ret = torch.tensor([a_on[i]/total for i in range(10)]).view(1,-1)
    b_ret = torch.tensor([b_on[i]/total for i in range(10)]).view(1,-1)
    c_ret = torch.tensor([c_on[i]/total for i in range(10)]).view(1,-1)
  else:
    a_ret, b_ret, c_ret = torch.zeros((1,10)), torch.zeros((1,10)), torch.zeros((1,10))

  return (a_ret, b_ret, c_ret)

class MNISTSum3Dataset(torch.utils.data.Dataset):
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
    digits = [item[3] for item in batch]
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

  def sum_3_test(self, digit_1, digit_2, digit_3):
    (dim, _) = digit_1.shape
    a_cat = Categorical(probs=digit_1)
    b_cat = Categorical(probs=digit_2)
    c_cat = Categorical(probs=digit_3)
    a_samples = torch.t(a_cat.sample((args.n_samples,)))
    b_samples = torch.t(b_cat.sample((args.n_samples,)))
    c_samples = torch.t(c_cat.sample((args.n_samples,)))
    results = torch.zeros(dim)
    for i in range(dim):
      results[i] = torch.mode(a_samples[i] + b_samples[i] + c_samples[i]).values.item()
    return results

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor], y: List[int]):
    """
    Invoked during training

    Takes in input pair of images (x_a, x_b) and the ground truth output sum (r)
    Returns the loss (a single scalar)
    """
    (a_imgs, b_imgs, c_imgs) = x

    # First recognize the two digits
    a_distrs = self.mnist_net(a_imgs) # Tensor 64 x 10
    b_distrs = self.mnist_net(b_imgs) # Tensor 64 x 10
    c_distrs = self.mnist_net(c_imgs) # Tensor 64 x 10

    a_distrs_list = list(a_distrs.clone().detach())
    b_distrs_list = list(b_distrs.clone().detach())
    c_distrs_list = list(c_distrs.clone().detach())
    
    argss = list(zip(a_distrs_list, b_distrs_list, c_distrs_list, y))
    out_pred = map(sample_fn, argss)
    out_pred = list(zip(*out_pred))

    a_pred, b_pred, c_pred = out_pred[0], out_pred[1], out_pred[2]
    a_pred = torch.stack(a_pred).view([a_distrs.shape[0],10])
    b_pred = torch.stack(b_pred).view([b_distrs.shape[0],10])
    c_pred = torch.stack(c_pred).view([c_distrs.shape[0],10])

    cat_distrs = torch.cat((a_distrs, b_distrs, c_distrs))
    cat_pred = torch.cat((a_pred, b_pred, c_pred))
    l = F.mse_loss(cat_distrs,cat_pred)

    return l

  def evaluate(self, x: Tuple[torch.Tensor, torch.Tensor]):
    """
    Invoked during testing

    Takes in input pair of images (x_a, x_b)
    Returns the predicted sum of the two digits (vector of dimension 19)
    """
    (a_imgs, b_imgs, c_imgs) = x

    # First recognize the two digits
    a_distrs = self.mnist_net(a_imgs) # Tensor 64 x 10
    b_distrs = self.mnist_net(b_imgs) # Tensor 64 x 10
    c_distrs = self.mnist_net(c_imgs) # Tensor 64 x 10

    return self.sum_3_test(digit_1=a_distrs, digit_2=b_distrs, digit_3=c_distrs) # Tensor 64 x 19

class Trainer():
  def __init__(self, train_loader, test_loader, learning_rate, k, provenance):
    self.network = MNISTSum3Net(provenance, k)
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
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sum_3_inv")
  parser.add_argument("--n-epochs", type=int, default=10)
  parser.add_argument("--batch-size-train", type=int, default=64)
  parser.add_argument("--batch-size-test", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--top-k", type=int, default=3)
  parser.add_argument("--jit", action="store_true")
  parser.add_argument("--n-samples", type=int, default=100)
  parser.add_argument("--dispatch", type=str, default="parallel")
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size_train = args.batch_size_train
  batch_size_test = args.batch_size_test
  learning_rate = args.learning_rate
  k = args.top_k
  provenance = args.provenance
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))

  # Dataloaders
  train_loader, test_loader = mnist_sum_3_loader(data_dir, batch_size_train, batch_size_test)

  # Create trainer and train
  trainer = Trainer(train_loader, test_loader, learning_rate, k, provenance)
  trainer.train(n_epochs)
