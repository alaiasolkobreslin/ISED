import os
import json
from typing import *
import random
from argparse import ArgumentParser
from tqdm import tqdm
import math

import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from PIL import Image

import common

def hwf(expr):
  n = len(expr)
  for i in range(n):
      if i % 2 == 0 and not expr[i].isdigit():
          raise Exception("Invalid HWF")
      elif i % 2 == 1 and expr[i] not in ['+', '*', '-', '/']:
          raise Exception("Invalid HWF")
  return eval(expr)


class HWFDataset(torch.utils.data.Dataset):
  def __init__(self, root: str, prefix: str, split: str):
    super(HWFDataset, self).__init__()
    self.root = root
    self.split = split
    self.metadata = json.load(open(os.path.join(root, f"HWF/{prefix}_{split}.json")))
    self.img_transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5,), (1,))])

  def __getitem__(self, index):
    sample = self.metadata[index]

    # Input is a sequence of images
    img_seq = []
    for img_path in sample["img_paths"]:
      img_full_path = os.path.join(self.root, "HWF/Handwritten_Math_Symbols", img_path)
      img = Image.open(img_full_path).convert("L")
      img = self.img_transform(img)
      img_seq.append(img)
    img_seq_len = len(img_seq)

    # Output is the "res" in the sample of metadata
    res = sample["res"]

    # Return (input, output) pair
    return (img_seq, img_seq_len, res)

  def __len__(self):
    return len(self.metadata)

  @staticmethod
  def collate_fn(batch):
    max_len = max([img_seq_len for (_, img_seq_len, _) in batch])
    zero_img = torch.zeros_like(batch[0][0][0])
    pad_zero = lambda img_seq: img_seq + [zero_img] * (max_len - len(img_seq))
    img_seqs = torch.stack([torch.stack(pad_zero(img_seq)) for (img_seq, _, _) in batch])
    img_seq_len = torch.stack([torch.tensor(img_seq_len).long() for (_, img_seq_len, _) in batch])
    results = torch.stack([torch.tensor(res) for (_, _, res) in batch])
    return (img_seqs, img_seq_len, results)


def hwf_loader(data_dir, batch_size, prefix):
  train_dataset = HWFDataset(data_dir, prefix, "train")
  train_set_size = len(train_dataset)
  train_indices = list(range(train_set_size))
  split = int(train_set_size * 0.8)
  train_indices, val_indices = train_indices[:split], train_indices[split:]
  train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, train_indices), collate_fn=HWFDataset.collate_fn, batch_size=batch_size, shuffle=True)
  valid_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, val_indices), collate_fn=HWFDataset.collate_fn, batch_size=batch_size, shuffle=True)
  
  test_loader = torch.utils.data.DataLoader(HWFDataset(data_dir, prefix, "test"), collate_fn=HWFDataset.collate_fn, batch_size=batch_size, shuffle=True)
  return (train_loader, valid_loader, test_loader)


class SymbolNet(nn.Module):
  def __init__(self):
    super(SymbolNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, stride = 1, padding = 1)
    self.conv2 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
    self.fc1 = nn.Linear(30976, 128)
    self.fc2 = nn.Linear(128, 14)

  def forward(self, x):
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


def hwf_eval(symbols: List[str]):
  # Sanitize the input
  for i, s in enumerate(symbols):
    if i % 2 == 0 and not s.isdigit(): raise Exception("BAD")
    if i % 2 == 1 and s not in ["+", "-", "*", "/"]: raise Exception("BAD")

  # Evaluate the result
  result = eval("".join(symbols))

  return result


class HWFNet(nn.Module):
  def __init__(self):
    super(HWFNet, self).__init__()
    self.symbol_cnn = SymbolNet()

  def forward(self, img_seq, img_seq_len):
    batch_size, formula_length, _, _, _ = img_seq.shape
    length = [l.item() for l in img_seq_len]
    symbol = self.symbol_cnn(img_seq.flatten(start_dim=0, end_dim=1)).view(batch_size, formula_length, -1)
    return (symbol, length)
  
class RLHWFNet(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    self.saved_log_probs = []
    self.rewards = []
    self.perception = HWFNet()

  def forward(self, x):
    return self.perception.forward(x)

def validation(symbols, lengths):
    a = a.argmax(dim=1)
    b = b.argmax(dim=1)
    sym = symbols.argmax(dim=2)
    
    predictions = torch.stack([torch.tensor(hwf_eval(sym[i][:lengths[i]])) for i in range(len(sym))])
    return predictions
  
def final_output(model,ground_truth, args, symbols, lengths):
  d_symbols = [torch.distributions.categorical.Categorical(s) for s in symbols]
  s_symbols = [d.sample() for d in d_symbols]
  
  model.saved_log_probs = sum([d.log_prob(s) for d, s in zip(d_symbols, s_symbols)])

  predictions = []
  for i in range(len(s_symbols)):
    prediction = hwf(s_symbols[i])
    predictions.append(prediction)
    reward = common.compute_reward(prediction,ground_truth[i])
    model.rewards.append(reward)
  
  return torch.stack(predictions)


if __name__ == "__main__":
  # Command line arguments
  parser = ArgumentParser("hwf")
  parser.add_argument('--gpu-id', default='cuda:0', type=str)
  parser.add_argument('-j', '--workers', default=0, type=int)
  parser.add_argument('--print-freq', default=5, type=int)
  parser.add_argument('--seed', default=1234, type=int)
  parser.add_argument("--dataset-prefix", type=str, default="expr")

  parser.add_argument('--epochs', default=20, type=int)
  parser.add_argument('--warmup', default=10, type=int)
  parser.add_argument('-b', '--batch-size', default=16, type=int)
  parser.add_argument('--learning-rate', default=0.0001, type=float)
  parser.add_argument('--weight-decay', default=3e-1, type=float)
  parser.add_argument('--clip-grad-norm', default=1., type=float)
  parser.add_argument('--disable-cos', action='store_true')
  parser.add_argument('--cuda', default=True, type=bool)
  args = parser.parse_args()

  # Parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  if args.cuda:
    if torch.cuda.is_available(): device = torch.device(args.gpu_id)
    else: raise Exception("No cuda available")
  else: device = torch.device("cpu")

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../../generation-pipeline/data/"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../model/hwf"))
  if not os.path.exists(model_dir): os.makedirs(model_dir)

  model = RLHWFNet()
  model.to(args.gpu_id)

  (train_loader, valid_loader, test_loader) = hwf_loader(data_dir, args.batch_size, args.dataset_prefix)
  trainer = common.Trainer(train_loader, valid_loader, test_loader, model, model_dir, final_output, args)
  trainer.train(args.epochs)