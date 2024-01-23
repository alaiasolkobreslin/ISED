from typing import Optional, Callable
import os
import random

import csv
import time

import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from PIL import Image

from argparse import ArgumentParser
from tqdm import tqdm

import leaves_config

leaves_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class LeavesDataset(torch.utils.data.Dataset):
  def __init__(
    self,
    data_root: str,
    data_dir: str,
    n_train: int,
    transform: Optional[Callable] = leaves_img_transform,
  ):
    self.transform = transform
    if data_dir == 'leaf_10': self.labels = leaves_config.l10_labels
    elif data_dir == 'leaf_30': self.labels = leaves_config.l30_labels
    elif data_dir == 'leaf_11': self.labels = leaves_config.l11_labels
    else: self.labels = []
    
    # Get all image paths and their labels
    self.samples = []
    data_dir = os.path.join(data_root, data_dir)
    data_dirs = os.listdir(data_dir)
    for sample_group in data_dirs:
      sample_group_dir = os.path.join(data_dir, sample_group)
      if not os.path.isdir(sample_group_dir) or not sample_group in self.labels:
        continue
      label = self.labels.index(sample_group)
      sample_group_files = os.listdir(sample_group_dir)
      for idx in random.sample(range(len(sample_group_files)), min(n_train, len(sample_group_files))):
        sample_img_path = os.path.join(sample_group_dir, sample_group_files[idx])
        if sample_img_path.endswith('png') or sample_img_path.endswith('JPG'):
          self.samples.append((sample_img_path, label))
    
    self.index_map = list(range(len(self.samples)))
    random.shuffle(self.index_map)

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    (img_path, label) = self.samples[self.index_map[idx]]
    img = Image.open(open(img_path, "rb"))
    img = self.transform(img)
    return (img, label)
  
  @staticmethod
  def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.stack([torch.tensor(item[1]).long() for item in batch])
    return (imgs, labels)

def leaves_loader(data_root, data_dir, n_train, batch_size, train_percentage):
  dataset = LeavesDataset(data_root, data_dir, n_train)
  num_train = int(len(dataset) * train_percentage)
  num_test = len(dataset) - num_train
  (train_dataset, test_dataset) = torch.utils.data.random_split(dataset, [num_train, num_test])
  train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=LeavesDataset.collate_fn, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=LeavesDataset.collate_fn, batch_size=batch_size, shuffle=True)
  return (train_loader, test_loader)

class LeavesNet(nn.Module):
  def __init__(self, data_dir):
    super(LeavesNet, self).__init__()
    if data_dir == 'leaf_11':
      self.num_classes = 11
      self.dim = 2304
    elif data_dir == 'leaf_10':
      self.num_classes = 10
      self.dim = 3072
    elif data_dir == 'leaf_30':
      self.num_classes = 30
      self.dim = 3072
    else:
      raise Exception(f"Unknown directory: {data_dir}")
  
    # CNN
    self.cnn = nn.Sequential(
      nn.Conv2d(3, 32, 10, 1),
      nn.ReLU(),
      nn.MaxPool2d(3),
      nn.Conv2d(32, 64, 5, 1),
      nn.ReLU(),
      nn.MaxPool2d(3),
      nn.Conv2d(64, 128, 3, 1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(128, 128, 3, 1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Flatten(),
    )

    self.last_fc = nn.Sequential(
      nn.Linear(self.dim, self.dim),
      nn.ReLU(),
      nn.Dropout(),
      nn.Linear(self.dim, self.num_classes),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    x = self.cnn(x)
    x = self.last_fc(x)
    return x

class Trainer():
  def __init__(self, train_loader, test_loader, learning_rate, data_dir, gpu, save_model=False):
    if gpu >= 0:
      device = torch.device("cuda:%d" % gpu)
    else:
      device = torch.device("cpu")
    self.device = device
    self.network = LeavesNet(data_dir) #.to(self.device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.save_model = save_model
    self.min_test_loss = 100000000.0

  def loss_fn(self, output, ground_truth):
    dim = output.shape[1]
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
    return F.binary_cross_entropy(output, gt)
  
  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    total_correct = 0
    train_loss = 0
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for (i, (input, target)) in enumerate(iter):
      self.optimizer.zero_grad()
      input = input.to(self.device)
      target = target.to(self.device)
      output = self.network(input)
      loss = self.loss_fn(output, target)
      loss.backward()
      self.optimizer.step()
      total_correct += (output.argmax(dim=1)==target).float().sum()
      num_items += output.shape[0]
      correct_perc = 100. * total_correct / num_items
      train_loss += loss.item()
      avg_loss = train_loss / (i + 1)
      iter.set_description(f"[Train Epoch {epoch}] Avg Loss: {avg_loss:.4f}, Overall Accuracy: {int(total_correct)}/{int(num_items)} ({correct_perc:.2f})%")

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = 0
    num_correct = 0
    test_loss = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for i, (input, target) in enumerate(iter):
        input = input.to(self.device)
        target = target.to(self.device)
        output = self.network(input)
        num_items += output.shape[0]
        test_loss += self.loss_fn(output, target).item()
        num_correct += (output.argmax(dim=1)==target).float().sum()
        perc = 100.*num_correct/num_items
        avg_loss = test_loss / (i + 1)
        iter.set_description(f"[Test Epoch {epoch}] Avg loss: {avg_loss:.4f}, Accuracy: {int(num_correct)}/{int(num_items)} ({perc:.2f})%")
    
    # if self.save_model and test_loss < self.min_test_loss:
    #  self.min_test_loss = test_loss
    #  torch.save(self.network, "../model/leaves/leaves_net.pkl")
    return float(num_correct/num_items)
  
  def train(self, n_epochs):
    dict = {}
    for epoch in range(1, n_epochs+1):
      t0 = time.time()
      self.train_epoch(epoch)
      t1 = time.time()
      dict["time epoch " + str(epoch)] = round(t1 - t0, ndigits=4)
      acc = self.test_epoch(epoch)
      dict["accuracy epoch " + str(epoch)] = round(acc, ndigits=6)
    return dict

if __name__ == "__main__":
  parser = ArgumentParser("leaves")
  parser.add_argument("--model-name", type=str, default="leaves.pkl")
  parser.add_argument("--n-epochs", type=int, default=30)
  parser.add_argument("--gpu", type=int, default=-1)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--cuda", action="store_true")
  args = parser.parse_args()

  random_seeds = [1234, 3177, 5848, 9175, 8725]
  train_nums = [20, 35, 45, 65, 115]
  train_percentages = [0.5, 0.6, 0.7, 0.8, 0.9]
  data_dirs = ['leaf_11']
  accuracies = ["accuracy epoch " + str(i+1) for i in range(args.n_epochs)]
  times = ["time epoch " + str(i+1) for i in range(args.n_epochs)]
  field_names = ['random seed', 'data_dir', 'num train'] + accuracies + times

  with open('demo/leaf/leaf_baseline.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()
    csvfile.close()

  for data_dir in data_dirs:
    for i in range(len(train_nums)): # 10, 20, 30, 50, 100
      for seed in random_seeds:
        torch.manual_seed(seed)
        random.seed(seed)
        
        data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../benchmarks/data"))
        model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/leaves"))
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        
        (train_loader, test_loader) = leaves_loader(data_root, data_dir, train_nums[i], args.batch_size, train_percentages[i])
        trainer = Trainer(train_loader, test_loader, args.learning_rate, data_dir, args.gpu)

        dict = trainer.train(args.n_epochs)
        dict["random seed"] = seed
        dict['data_dir'] = data_dir
        dict["num train"] = int(train_percentages[i]*train_nums[i])
        with open('demo/leaf/leaf_baseline.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writerow(dict)
            csvfile.close() 