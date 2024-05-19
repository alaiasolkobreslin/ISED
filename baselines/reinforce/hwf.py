import os
import random
from typing import *
import json
from PIL import Image
import csv
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from argparse import ArgumentParser

class HWFDataset(torch.utils.data.Dataset):
  def __init__(self, root: str, prefix: str, split: str):
    super(HWFDataset, self).__init__()
    self.root = root
    self.split = split
    self.metadata = json.load(open(os.path.join(root, f"HWF/{prefix}_{split}.json")))
    self.img_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (1,))
      ])
    
  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, index):
    sample = self.metadata[index]

    # Input is a sequence of images
    img_seq = []
    for img_path in sample["img_paths"]:
      img_full_path = os.path.join(self.root, "HWF/Handwritten_Math_Symbols", img_path)
      img = Image.open(img_full_path).convert("L")
      img = self.img_transform(img)
      img_seq.append(img)

    # Output is the "res" in the sample of metadata
    res = sample["res"]
    img_seq_len = len(img_seq)

    # Return (input, output) pair
    return (img_seq, img_seq_len, res)

  @staticmethod
  def collate_fn(batch):
    max_len = 7
    zero_img = torch.zeros_like(batch[0][0][0])
    pad_zero = lambda img_seq: img_seq + [zero_img] * (max_len - len(img_seq))
    img_seqs = torch.stack([torch.stack(pad_zero(img_seq)) for (img_seq, _, _) in batch])
    img_seq_len = torch.stack([torch.tensor(img_seq_len).long() for (_, img_seq_len, _) in batch])
    results = torch.stack([torch.tensor(res) for (_, _, res) in batch])
    return (img_seqs, img_seq_len, results)

def hwf_loader(data_dir, batch_size):
  prefix = "expr"
  train_loader = torch.utils.data.DataLoader(HWFDataset(data_dir, prefix, "train"), collate_fn=HWFDataset.collate_fn, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(HWFDataset(data_dir, prefix, "test"), collate_fn=HWFDataset.collate_fn, batch_size=batch_size, shuffle=True)
  return (train_loader, test_loader)

class SymbolNet(nn.Module):
    def __init__(self):
        super(SymbolNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(30976, 128)
        self.fc2 = nn.Linear(128, 14)

    def forward_symbol(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, p=0.25, training=self.training)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def forward(self, x):
      symbol = self.forward_symbol(x)
      return symbol
  
def eval_result_eq(a, b, threshold=0.01):
    result = abs(a - b) < threshold
    return result

def loss_fn(data, len, target, task):
    symbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', "+", "-", "*", "/"]
    acc = []
    dim = data.shape[:-1].numel()
    data = data.flatten(0,-2)
    for d in range(dim):
      ind = 0
      for n in range(target.shape[0]):
        i = data[d][ind:ind+len[n]]
        ind += len[n]
        input = [symbols[int(j)] for j in i]
        try:
          y = task(input)
          acc.append(eval_result_eq(y, target[n]).float())
        except:
          acc.append(torch.tensor(0.))
    acc = torch.stack(acc).reshape(dim, -1)
    return acc

class Trainer():
  def __init__(self, model, loss_fn, train_loader, test_loader, model_dir, learning_rate, grad_type, dim, sample_count, seed, task):
    self.model_dir = model_dir
    self.network = model()
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = None
    self.grad_type = grad_type
    self.dim = dim
    self.sample_count = sample_count
    self.loss_fn = loss_fn
    self.seed = seed
    self.task = task

  def indecater_multiplier(self, batch_size):
    icr_mult = torch.zeros((self.dim, 14, self.sample_count, batch_size, self.dim))
    icr_replacement = torch.zeros((self.dim, 14, self.sample_count, batch_size, self.dim))
    for i in range(self.dim):
      for j in range(14):
        icr_mult[i,j,:,:,i] = 1
        icr_replacement[i,j,:,:,i] = j
    return icr_mult, icr_replacement

  def reinforce_grads(self, data, eqn_len, target):
    data = torch.cat([d[:eqn_len[i]] for i, d in enumerate(data)], dim=0)
    logits = self.network(data)
    d = torch.distributions.Categorical(logits=logits)
    samples = d.sample((self.sample_count,))
    f_sample = self.loss_fn(samples, eqn_len, target, self.task)
    f_mean = f_sample.mean(dim=0)

    log_p_sample = d.log_prob(samples)
    log_p_sample = [log_p_sample[:,eqn_len[:i].sum():eqn_len[:i+1].sum()].sum(dim=-1) for i, _ in enumerate(eqn_len)]
    log_p_sample = torch.stack(log_p_sample, dim=1)

    reinforce = (f_sample.detach() * log_p_sample).mean(dim=0)
    reinforce_prob = (f_mean - reinforce).detach() + reinforce
    loss = -torch.log(reinforce_prob + 1e-8)
    loss = loss.mean(dim=0)
    return loss
  
  def indecater_grads(self, data, eqn_len, target):
    batch_size = data.shape[0]
    data = torch.cat([d[:eqn_len[i]] for i, d in enumerate(data)], dim=0)
    logits = self.network(data)
    d = torch.distributions.Categorical(logits=logits)
    samples = d.sample((self.sample_count,))
    f_sample = self.loss_fn(samples, eqn_len, target, self.task)
    f_mean = f_sample.mean(dim=0)

    outer_samples = torch.stack([samples] * 14, dim=0)
    outer_samples = torch.stack([outer_samples] * self.dim, dim=0)
    m, r = self.indecater_multiplier(batch_size)
    m = torch.cat([m[:,:,:,i,:eqn_len[i]] for i in range(batch_size)], dim=-1)
    r = torch.cat([r[:,:,:,i,:eqn_len[i]] for i in range(batch_size)], dim=-1)
    outer_samples = outer_samples * (1 - m) + r
    outer_loss = self.loss_fn(outer_samples, eqn_len, target, self.task).reshape(self.dim, 14, self.sample_count, -1)
    variable_loss = outer_loss.mean(dim=2).permute(2,0,1)

    logits = [F.pad(F.softmax(logits[eqn_len[:i].sum():eqn_len[:i+1].sum(), :], dim=-1), (0,0,0,7-eqn_len[i])) for i, _ in enumerate(eqn_len)]
    logits = torch.stack(logits, dim=0)
    indecater_expression = variable_loss.detach() * logits
    indecater_expression = indecater_expression.sum(dim=-1)
    indecater_expression = indecater_expression.sum(dim=-1)

    icr_prob = (f_mean - indecater_expression).detach() + indecater_expression
    loss = -torch.log(indecater_expression + 1e-8) # -torch.log(icr_prob + 1e-8)
    loss = loss.mean(dim=0)
    return loss
  
  def grads(self, data, eqn_len, target):
    if self.grad_type == 'reinforce':
      return self.reinforce_grads(data, eqn_len, target)
    elif self.grad_type == 'icr':
      return self.indecater_grads(data, eqn_len, target)

  def train_epoch(self, epoch):
    train_loss = 0
    print(f"Epoch {epoch}")
    self.network.train()
    for (data, eqn_len, target) in self.train_loader:
      self.optimizer.zero_grad()
      loss = self.grads(data, eqn_len, target)
      train_loss += loss
      loss.backward()
      self.optimizer.step()
    return train_loss

  def test(self):
    num_items = len(self.test_loader.dataset)
    correct = 0
    with torch.no_grad():
      for (data, eqn_len, target) in self.test_loader:
        data = torch.cat([d[:eqn_len[i]] for i, d in enumerate(data)], dim=0)
        output = self.network(data)
        pred = self.loss_fn(output.argmax(dim=-1).unsqueeze(0), eqn_len, target, self.task)
        correct += pred.sum()
      
      perc = float(correct / num_items)
      if self.best_loss is None or self.best_loss < perc:
        self.best_loss = perc
        torch.save(self.network, model_dir+f"/{self.grad_type}_{self.seed}_best.pkl")

    return perc

  def train(self, n_epochs):
    dict = {}
    for epoch in range(1, n_epochs + 1):
      t0 = time.time()
      train_loss = self.train_epoch(epoch)
      t1 = time.time()
      acc = self.test()
      dict["L " + str(epoch)] = round(float(train_loss), ndigits=6)
      dict["A " + str(epoch)] = round(float(acc), ndigits=6)
      dict["T " + str(epoch)] = round(t1 - t0, ndigits=6)
      print(f"Test accuracy: {acc}")
    torch.save(self.network, model_dir+f"/{self.grad_type}_{self.seed}_last.pkl")
    return dict

def hwf(expr):
    n = len(expr)
    for i in range(n):
      if i % 2 == 0 and not expr[i].isdigit():
        raise Exception("Invalid HWF")
      elif i % 2 == 1 and expr[i] not in ['+', '*', '-', '/']:
        raise Exception("Invalid HWF")
    return eval("".join(expr))

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("hwf")
  parser.add_argument("--n-epochs", type=int, default=50)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--digit", type=int, default=7)
  parser.add_argument("--jit", action="store_true")
  parser.add_argument("--dispatch", type=str, default="parallel")
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  digits = args.digit

  accuracies = ["A " + str(i+1) for i in range(args.n_epochs)]
  times = ["T " + str(i+1) for i in range(args.n_epochs)]
  losses = ["L " + str(i+1) for i in range(args.n_epochs)]
  field_names = ['random seed', 'grad_type', 'task_type', 'sample count'] + accuracies + times + losses

  with open('baselines/reinforce/icr.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        csvfile.close()

  for grad_type in ['reinforce', 'icr']:
    for seed in [3177, 5848, 9175, 8725, 1234, 1357, 2468, 548, 6787, 8371]:
        torch.manual_seed(seed)
        random.seed(seed)

        if grad_type == 'reinforce': sample_count = 100
        elif grad_type == 'icr': sample_count = 2
        print(sample_count)
        print(seed)
        print(grad_type)

        # Data
        data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../../finite_diff/data"))
        model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f"../../model/hwf"))
        os.makedirs(model_dir, exist_ok=True)

        # Dataloaders
        train_loader, test_loader = hwf_loader(data_dir, batch_size)

        # Create trainer and train
        trainer = Trainer(SymbolNet, loss_fn, train_loader, test_loader, model_dir, learning_rate, grad_type, digits, sample_count, seed, hwf)
        dict = trainer.train(n_epochs)
        dict["random seed"] = seed
        dict['grad_type'] = grad_type
        dict['task_type'] = "hwf"
        dict['sample count'] = sample_count
        with open('baselines/reinforce/icr.csv', 'a', newline='') as csvfile:
          writer = csv.DictWriter(csvfile, fieldnames=field_names)
          writer.writerow(dict)
          csvfile.close()