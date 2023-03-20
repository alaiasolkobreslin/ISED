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


def sample_fn(arg):
  a_dist, b_dist, y = arg[0], arg[1], arg[2]

  a_cat, b_cat = Categorical(probs=a_dist), Categorical(probs=b_dist)
  a_samples, b_samples = a_cat.sample((args.n_samples,)), b_cat.sample((args.n_samples,))
  a_on, b_on = dict.fromkeys(range(10), 0), dict.fromkeys(range(10), 0)
  for i in range(args.n_samples):
    if a_samples[i] >= b_samples[i] == y:
      a_on[a_samples[i].item()] += 1
      b_on[b_samples[i].item()] += 1

  total = sum(a_on.values())
  
  if total:
    a_ret = torch.tensor([a_on[i]/total for i in range(10)]).view(1,-1)
    b_ret = torch.tensor([b_on[i]/total for i in range(10)]).view(1,-1)
  else:
    a_ret, b_ret = torch.zeros((1,10)), torch.zeros((1,10))

  return (a_ret, b_ret)

class MNISTSort2Dataset(torch.utils.data.Dataset):
  mnist_img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
      (0.1307,), (0.3081,)
    )
  ])

  def __init__(
    self,
    root: str,
    train: bool = True,
    target_transform: Optional[Callable] = None,
    download: bool = False,
    max_digit: Optional[int] = None,
  ):
    # Contains a MNIST dataset
    self.mnist_dataset = torchvision.datasets.MNIST(
      root,
      train=train,
      transform=self.mnist_img_transform,
      target_transform=target_transform,
      download=download,
    )
    self.index_map = list(range(len(self.mnist_dataset)))
    random.shuffle(self.index_map)

    # Check if we want limited labels
    if max_digit is not None:
      self.index_map = [i for i in self.index_map if self.mnist_dataset[i][1] <= max_digit]

  def __len__(self):
    return int(len(self.index_map) / 2)

  def __getitem__(self, idx):
    # Get two data points
    (a_img, a_digit) = self.mnist_dataset[self.index_map[idx * 2]]
    (b_img, b_digit) = self.mnist_dataset[self.index_map[idx * 2 + 1]]

    # Each data has two images and the GT is the comparison result of two digits
    return (a_img, b_img, 1 if a_digit < b_digit else 0)

  @staticmethod
  def collate_fn(batch):
    a_imgs = torch.stack([item[0] for item in batch])
    b_imgs = torch.stack([item[1] for item in batch])
    cmp = torch.stack([torch.tensor(item[2]).long() for item in batch])
    return ((a_imgs, b_imgs), cmp)


def mnist_sort_2_loader(data_dir, batch_size, max_digit):
  train_loader = torch.utils.data.DataLoader(
    MNISTSort2Dataset(
      data_dir,
      train=True,
      download=True,
      max_digit=max_digit,
    ),
    collate_fn=MNISTSort2Dataset.collate_fn,
    batch_size=batch_size,
    shuffle=True,
  )

  test_loader = torch.utils.data.DataLoader(
    MNISTSort2Dataset(
      data_dir,
      train=False,
      download=True,
      max_digit=max_digit,
    ),
    collate_fn=MNISTSort2Dataset.collate_fn,
    batch_size=batch_size,
    shuffle=True,
  )

  return train_loader, test_loader


class MNISTNet(nn.Module):
  def __init__(self, num_classes=10):
    super(MNISTNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
    self.fc1 = nn.Linear(1024, 1024)
    self.fc2 = nn.Linear(1024, num_classes)

  def forward(self, x):
    x = F.max_pool2d(self.conv1(x), 2)
    x = F.max_pool2d(self.conv2(x), 2)
    x = x.view(-1, 1024)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)
    return F.softmax(x, dim=1)


class MNISTSort2Net(nn.Module):
  def __init__(self, provenance, train_k, test_k, max_digit=9):
    super(MNISTSort2Net, self).__init__()
    self.max_digit = max_digit
    self.num_classes = max_digit + 1

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet(num_classes=self.num_classes)

  def less_than_test(self, digit_1, digit_2):
    (dim, _) = digit_1.shape
    a_cat, b_cat = Categorical(probs=digit_1), Categorical(probs=digit_2)
    a_samples = torch.t(a_cat.sample((args.n_samples,)))
    b_samples = torch.t(b_cat.sample((args.n_samples,)))
    results = torch.zeros(dim)
    for i in range(dim):
      results[i] = torch.mode(a_samples[i] < b_samples[i]).values.item()
    return results  

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor], y: List[int]):
    (a_imgs, b_imgs) = x

    # First recognize the two digits
    a_distrs = self.mnist_net(a_imgs) # Tensor 64 x 10
    b_distrs = self.mnist_net(b_imgs) # Tensor 64 x 10

    a_distrs_list = list(a_distrs.clone().detach())
    b_distrs_list = list(b_distrs.clone().detach())
    argss = list(zip(a_distrs_list, b_distrs_list, y))
    out_pred = map(sample_fn, argss)
    out_pred = list(zip(*out_pred))

    a_pred, b_pred = out_pred[0], out_pred[1]
    a_pred = torch.stack(a_pred).view([a_distrs.shape[0],10])
    b_pred = torch.stack(b_pred).view([b_distrs.shape[0],10])

    cat_distrs = torch.cat((a_distrs, b_distrs))
    cat_pred = torch.cat((a_pred, b_pred))
    l = F.mse_loss(cat_distrs,cat_pred)
    return l

  def evaluate(self, x: Tuple[torch.Tensor, torch.Tensor]):
    """
    Invoked during testing

    Takes in input pair of images (x_a, x_b)
    Returns the predicted sum of the two digits (vector of dimension 19)
    """
    (a_imgs, b_imgs) = x

    # First recognize the two digits
    a_distrs = self.mnist_net(a_imgs) # Tensor 64 x 10
    b_distrs = self.mnist_net(b_imgs) # Tensor 64 x 10

    return self.less_than_test(digit_1=a_distrs, digit_2=b_distrs) # Tensor 64 x 19


class Trainer():
  def __init__(self, train_loader, test_loader, learning_rate, loss, train_k, test_k, provenance, max_digit=9):
    self.network = MNISTSort2Net(provenance, train_k=train_k, test_k=test_k, max_digit=max_digit)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.min_test_loss = 100000000.0

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
    self.test_epoch(0)
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sort_2")
  parser.add_argument("--n-epochs", type=int, default=10)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--train-k", type=int, default=3)
  parser.add_argument("--test-k", type=int, default=3)
  parser.add_argument("--n-samples", type=int, default=100)
  parser.add_argument("--wmc-type", type=str, default="bottom-up")
  parser.add_argument("--max-digit", type=int, default=9)
  args = parser.parse_args()

  # Parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))

  # Dataloaders
  train_loader, test_loader = mnist_sort_2_loader(data_dir, args.batch_size, max_digit=args.max_digit)

  # Create trainer and train
  trainer = Trainer(train_loader, test_loader, args.learning_rate, args.loss_fn, args.train_k, args.test_k, args.provenance, max_digit=args.max_digit)
  trainer.train(args.n_epochs)
