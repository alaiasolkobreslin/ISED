import os
import json
import random
from argparse import ArgumentParser
from tqdm import tqdm
import math
from typing import *

import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from PIL import Image

import math

from util import sample

def is_valid(formula):
   digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
   for i in range(len(formula)):
      if i % 2 == 0 and formula[i] not in digits:
         return False
      if i % 2 == 1 and formula[i] in digits:
         return False
   return True

def evaluate_formula(formula):
    if not is_valid(formula):
       return 0 # TODO: FIX THIS
    """Evaluates a hand-written formula given as a list of characters in infix notation."""
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    num_stack = []
    op_stack = []
    for c in formula:
        if c.isdigit():
            num_stack.append(int(c))
        elif c in {'+', '-', '*', '/'}:
            while op_stack and precedence[c] <= precedence[op_stack[-1]]:
                op = op_stack.pop()
                b = num_stack.pop()
                a = num_stack.pop()
                if op == '+':
                    num_stack.append(a + b)
                elif op == '-':
                    num_stack.append(a - b)
                elif op == '*':
                    num_stack.append(a * b)
                elif op == '/':
                    if b != 0:
                      num_stack.append(a / b)
                    else:
                       return 0 # TODO: FIX THIS
            op_stack.append(c)
        else:
            return 0 # TODO: FIX THIS
    while op_stack:
        op = op_stack.pop()
        b = num_stack.pop()
        a = num_stack.pop()
        if op == '+':
            num_stack.append(a + b)
        elif op == '-':
            num_stack.append(a - b)
        elif op == '*':
            num_stack.append(a * b)
        elif op == '/':
            if b != 0:
              num_stack.append(a / b)
            else:
               return 0 # TODO: FIX THIS
    if len(num_stack) != 1:
        return 0 # TODO: FIX THIS
    return num_stack[0]

def idx_to_char(idx):
   if idx < 10:
      return str(idx)
   elif idx == 10:
      return "+"
   elif idx == 11:
      return "-"
   elif idx == 12:
      return "*"
   elif idx == 13:
      return "/"

def hwf_forward(inputs, length):
   # ["+", "-", "*", "/"]
   real_inputs = inputs[:length.item()]
   samples = real_inputs[0].shape[0]
   result = torch.zeros(samples)
   for i in range(samples):
      formula = [idx_to_char(input[i].item()) for input in real_inputs]
      result[i] = evaluate_formula(formula)
   return result

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
  train_loader = torch.utils.data.DataLoader(HWFDataset(data_dir, prefix, "train"), collate_fn=HWFDataset.collate_fn, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(HWFDataset(data_dir, prefix, "test"), collate_fn=HWFDataset.collate_fn, batch_size=batch_size, shuffle=True)
  return (train_loader, test_loader)


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


class HWFNet(nn.Module):
  def __init__(self, n_symbols):
    super(HWFNet, self).__init__()

    # Symbol embedding
    self.symbol_cnn = SymbolNet()
    self.n_symbols = n_symbols
    input_mapping = [14] * n_symbols
    self.sampling = sample.Sample(n_inputs=n_symbols, n_samples=args.n_samples, input_mapping=input_mapping, fn=hwf_forward)

  def hwf_test(self, symbols, img_seq_len):
     return self.sampling.sample_test(symbols, img_seq_len)

  def forward(self, img_seq, img_seq_len, y:List[float]):
    batch_size, n_symbols, _, _, _ = img_seq.shape
    t = torch.transpose(img_seq, 0, 1)
    imgs = [t[i] for i in range(n_symbols)]
    
    batched = torch.cat(tuple(imgs))
    distrs = self.symbol_cnn(batched).split(batch_size)
    distrs_list = [distr.clone().detach() for distr in distrs]
    
    argss = list(zip(*(tuple(distrs_list)), y))
    out_pred = map(self.sampling.sample_train, argss, list(img_seq_len))
    out_pred = list(zip(*out_pred))
    preds = [torch.stack(out_pred[i]).view([distrs[i].shape[0], 14]) for i in range(n_symbols)]

    cat_distrs = torch.cat(distrs)
    cat_pred = torch.cat(preds)
    l = F.mse_loss(cat_distrs,cat_pred)
    return l

  def evaluate(self, img_seq, img_seq_len):
    batch_size, n_symbols, _, _, _ = img_seq.shape
    t = torch.transpose(img_seq, 0, 1)
    imgs = [t[i] for i in range(n_symbols)]

    batched = torch.cat(tuple(imgs))
    distrs = self.symbol_cnn(batched).split(batch_size)

    return self.hwf_test(distrs, img_seq_len)

class Trainer():
  def __init__(self, train_loader, test_loader, device, model_root, model_name, learning_rate, n_symbols):
    self.network = HWFNet(n_symbols).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.device = device
    self.loss_fn = F.binary_cross_entropy
    self.model_root = model_root
    self.model_name = model_name
    self.min_test_loss = 100000000.0

  def eval_result_eq(self, a, b, threshold=0.01):
    result = abs(a - b) < threshold
    return result

  def train_epoch(self, epoch):
    self.network.train()
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    total_loss = 0.0
    for (batch_id, (img_seq, img_seq_len, target)) in enumerate(iter):
      self.optimizer.zero_grad()
      loss = self.network.forward(img_seq.to(device), img_seq_len.to(device), target)
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
      for (img_seq, img_seq_len, label) in iter:
        batch_size = len(label)
        output = self.network.evaluate(img_seq.to(device), img_seq_len.to(device))

        for i in range(batch_size):
          if output[i].item() == label[i]:
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
  # Command line arguments
  parser = ArgumentParser("hwf")
  parser.add_argument("--model-name", type=str, default="hwf.pkl")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--dataset-prefix", type=str, default="expr")
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--cuda", action="store_true")
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--n-samples", type=int, default=100)
  parser.add_argument("--difficulty", type=str, default="easy")
  args = parser.parse_args()

  # Read json
  dir_path = os.path.dirname(os.path.realpath(__file__))
  data = json.load(open(os.path.join(dir_path ,os.path.join('specs', 'hwf.json'))))

  # Parameters
  n_symbols = data['n_symbols'][args.difficulty]
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  if args.cuda:
    if torch.cuda.is_available(): device = torch.device(f"cuda:{args.gpu}")
    else: raise Exception("No cuda available")
  else: device = torch.device("cpu")

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/hwf"))
  if not os.path.exists(model_dir): os.makedirs(model_dir)
  train_loader, test_loader = hwf_loader(data_dir, batch_size=args.batch_size, prefix=args.dataset_prefix)

  # Training
  trainer = Trainer(train_loader, test_loader, device, model_dir, args.model_name, args.learning_rate, n_symbols)
  trainer.train(args.n_epochs)
