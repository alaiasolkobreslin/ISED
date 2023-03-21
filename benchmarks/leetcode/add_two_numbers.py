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

from util import sample

# https://leetcode.com/problems/add-two-numbers/

class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def add_two_numbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    carry = 0
    result = ListNode(0)
    pointer = result
      
    while (l1 or l2 or carry): 
        first_num = l1.val if l1.val else 0
        second_num = l2.val if l2.val else 0
        summation = first_num + second_num + carry
        num = summation % 10
        carry = summation // 10
        pointer.next = ListNode(num)
        pointer = pointer.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
        
    return result.next

def add_two_numbers_forward(inputs):
  return inputs[0] + inputs[1]

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
    digits = [item[2] for item in batch]
    return ((a_imgs, b_imgs), digits)


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
  def __init__(self):
    super(MNISTAddTwoNumbersNet, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()

    self.sampling = sample.Sample(n_inputs=2, n_samples=args.n_samples, input_mapping=[10, 10], fn=add_two_numbers_forward)

  def add_two_numbers_test(self, digit_1, digit_2):
    input_distrs = [digit_1, digit_2]
    return self.sampling.sample_test(input_distrs)

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor], y: List[int]):
    """
    Invoked during training

    Takes in input pair of images (x_a, x_b) and the ground truth output sum (r)
    Returns the loss (a single scalar)
    """
    (a_imgs, b_imgs) = x

    # First recognize the two digits
    a_distrs = self.mnist_net(a_imgs) # Tensor 64 x 10
    b_distrs = self.mnist_net(b_imgs) # Tensor 64 x 10

    a_distrs_list = list(a_distrs.clone().detach())
    b_distrs_list = list(b_distrs.clone().detach())
    argss = list(zip(a_distrs_list, b_distrs_list, y))
    out_pred = map(self.sampling.sample_train, argss)
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

    # Testing: execute the reasoning module; the result is a size 19 tensor
    return self.add_two_numbers_test(digit_1=a_distrs, digit_2=b_distrs) # Tensor 64 x 19


class Trainer():
  def __init__(self, train_loader, test_loader, learning_rate):
    self.network = MNISTAddTwoNumbersNet()
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
    self.test_epoch(0)
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

  # Parameters
  n_epochs = args.n_epochs
  batch_size_train = args.batch_size_train
  batch_size_test = args.batch_size_test
  learning_rate = args.learning_rate
  n_digits = 3
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))

  # Dataloaders
  train_loader, test_loader = MNIST_add_two_numbers_loader(data_dir, n_digits, batch_size_train, batch_size_test)

  # Create trainer and train
  trainer = Trainer(train_loader, test_loader, learning_rate)
  trainer.train(n_epochs)