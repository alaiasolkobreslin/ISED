import os
import random
from typing import *
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from argparse import ArgumentParser

def sum_m(*inputs):
    return sum(inputs)

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
    task: Callable,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False
  ):
    # Contains a MNIST dataset
    if train: self.length = min(5000 * digit, 60000)
    else: self.length = min(500 * digit, 10000)
    self.digit = digit
    self.task = task
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
    
    target = self.task(*tuple(target))

    # Each data has two images and the GT is the sum of two digits
    return (*tuple(data), target)

  @staticmethod
  def collate_fn(batch):
    imgs = []
    for i in range(len(batch[0])-1):
      imgs.append(torch.stack([item[i] for item in batch]))
    digits = torch.stack([torch.tensor(item[-1]).long() for item in batch])
    return (tuple(imgs), digits)

def mnist_digits_loader(data_dir, batch_size, digit, task):
  train_loader = torch.utils.data.DataLoader(
    MNISTDigitsDataset(
      data_dir,
      digit,
      task,
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
      task,
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
    return x

class MNISTTaskNet(nn.Module):
  def __init__(self, dim):
    super(MNISTTaskNet, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()
    self.dim = dim

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
    batch_size = x[0].shape[0]
    x = torch.cat(x, dim=0)
    x = self.mnist_net(x)
    x = torch.stack([x[i*batch_size:(i + 1) * batch_size,:] for i in range(self.dim)], dim=1)
    return x  

def sum_loss_fn(data, target, task):
    pred = data.sum(dim=-1)
    acc = torch.where(pred == target, 1., 0.)
    return acc

class Trainer():
  def __init__(self, model, loss_fn, train_loader, test_loader, model_dir, learning_rate, grad_type, dim, sample_count, task, task_type, seed):
    self.model_dir = model_dir
    self.network = model(dim)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = None
    self.grad_type = grad_type
    self.dim = dim
    self.sample_count = sample_count
    self.loss_fn = loss_fn
    self.task = task
    self.task_type = task_type
    self.seed = seed

  def indecater_multiplier(self, batch_size):
    icr_mult = torch.zeros((self.dim, 10, self.sample_count, batch_size, self.dim))
    icr_replacement = torch.zeros((self.dim, 10, self.sample_count, batch_size, self.dim))
    for i in range(self.dim):
      for j in range(10):
        icr_mult[i,j,:,:,i] = 1
        icr_replacement[i,j,:,:,i] = j
    return icr_mult, icr_replacement

  def reinforce_grads(self, data, target):
    logits = self.network(data)
    d = torch.distributions.Categorical(logits=logits)
    samples = d.sample((self.sample_count,))
    log_p_sample = d.log_prob(samples).sum(dim=-1) # digit sum

    f_sample = self.loss_fn(samples, target, self.task) # reward
    f_mean = f_sample.mean(dim=0)

    f_sample = (f_sample - f_mean)/(f_sample.std() + 1e-8)
    loss = -log_p_sample * f_sample
    loss = loss.mean()
    return loss
  
  def indecater_grads(self, data, target):
    logits = self.network(data)
    d = torch.distributions.Categorical(logits=logits)
    samples = d.sample((self.sample_count,))
    f_sample = self.loss_fn(samples, target.unsqueeze(0), self.task)
    f_mean = f_sample.mean(dim=0)
    batch_size = data[0].shape[0]

    outer_samples = torch.stack([samples] * 10, dim=0)
    outer_samples = torch.stack([outer_samples] * self.dim, dim=0)
    m, r = self.indecater_multiplier(batch_size)
    outer_samples = outer_samples * (1 - m) + r
    outer_loss = self.loss_fn(outer_samples, target.unsqueeze(0).unsqueeze(0).unsqueeze(0), self.task)
    
    variable_loss = outer_loss.mean(dim=2).permute(2,0,1)
    indecater_expression = variable_loss.detach() * F.softmax(logits, dim=-1)
    indecater_expression = indecater_expression.sum(dim=-1)
    indecater_expression = indecater_expression.sum(dim=-1)

    icr_prob = (f_mean - indecater_expression).detach() + indecater_expression
    loss = -torch.log(indecater_expression + 1e-8)
    loss = loss.mean(dim=0)
    return loss
  
  def advanced_indecater_grads(self, data, target):
    logits = self.network(data)
    d = torch.distributions.Categorical(logits=logits)
    samples = d.sample((self.sample_count * self.dim,))
    f_sample = self.loss_fn(samples, target.unsqueeze(0), self.task)
    f_mean = f_sample.mean(dim=0)
    batch_size = data[0].shape[0]

    samples = samples.reshape((self.dim, self.sample_count, batch_size, self.dim))
    outer_samples = torch.stack([samples] * 10, dim=1)
    m, r = self.indecater_multiplier(batch_size)
    outer_samples = outer_samples * (1 - m) + r
    outer_loss = self.loss_fn(outer_samples, target.unsqueeze(0).unsqueeze(0).unsqueeze(0), self.task)
    
    variable_loss = outer_loss.mean(dim=2).permute(2,0,1)
    indecater_expression = variable_loss.detach() * F.softmax(logits, dim=-1)
    indecater_expression = indecater_expression.sum(dim=-1)
    indecater_expression = indecater_expression.sum(dim=-1)

    icr_prob = (f_mean - indecater_expression).detach() + indecater_expression
    loss = -torch.log(indecater_expression + 1e-8)
    loss = loss.mean(dim=0)
    return loss
  
  def grads(self, data, target):
    if self.grad_type == 'reinforce':
      return self.reinforce_grads(data, target)
    elif self.grad_type == 'icr':
      return self.indecater_grads(data, target)
    elif self.grad_type == 'advanced_icr':
      return self.advanced_indecater_grads(data, target)

  def train_epoch(self, epoch):
    train_loss = 0
    print(f"Epoch {epoch}")
    self.network.train()
    for (data, target) in self.train_loader:
      self.optimizer.zero_grad()
      loss = self.grads(data, target)
      train_loss += loss
      loss.backward()
      self.optimizer.step()
    return train_loss

  def test(self):
    num_items = len(self.test_loader.dataset)
    correct = 0
    with torch.no_grad():
      for (data, target) in self.test_loader:
        output = self.network(data)
        acc = self.loss_fn(output.argmax(dim=-1), target, self.task)
        correct += acc.sum()
      perc = correct / num_items
    
    if self.best_loss is None or self.best_loss < perc:
      self.best_loss = perc
      torch.save(self.network, model_dir+f"/{self.grad_type}_{self.seed}_best.pkl")

    return perc

  def train(self, n_epochs):
    for epoch in range(1, n_epochs + 1):
      t0 = time.time()
      train_loss = self.train_epoch(epoch)
      t1 = time.time()
      acc = self.test()
      print(f"Test accuracy: {acc}")
    torch.save(self.network, model_dir+f"/{self.grad_type}_{self.seed}_last.pkl")

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_r")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--sample-count", type=int, default=100)
  parser.add_argument("--digits", type=int, default=16)
  parser.add_argument("--grad_type", type=str, default='reinforce')
  parser.add_argument("--jit", action="store_true")
  parser.add_argument("--dispatch", type=str, default="parallel")
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  sample_count = args.sample_count
  grad_type = args.grad_type
  digits = args.digits
  seed = args.seed

  torch.manual_seed(seed)
  random.seed(seed)
  task_type = f'sum_{digits}'
        
  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f"../../model/{task_type}"))
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  train_loader, test_loader = mnist_digits_loader(data_dir, batch_size, digits, sum_m)

  # Create trainer and train
  trainer = Trainer(MNISTTaskNet, sum_loss_fn, train_loader, test_loader, model_dir, learning_rate, grad_type, digits, sample_count, sum_m, task_type, seed)

  trainer.train(n_epochs)