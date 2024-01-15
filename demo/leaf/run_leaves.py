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

import blackbox
import leaves_config

leaves_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class LeavesDataset(torch.utils.data.Dataset):
  """
  :param data_root, the root directory of the data folder
  """
  def __init__(
    self,
    data_root: str,
    data_dir: str,
    n_train: int,
    transform: Optional[Callable] = leaves_img_transform,
  ):
    self.transform = transform
    
    # Get all image paths and their labels
    self.samples = []
    data_dir = os.path.join(data_root, data_dir)
    data_dirs = os.listdir(data_dir)
    for sample_group in data_dirs:
      sample_group_dir = os.path.join(data_dir, sample_group)
      if not os.path.isdir(sample_group_dir):
        continue
      label = leaves_config.l11_labels.index(sample_group)
      sample_group_files = os.listdir(sample_group_dir)
      for idx in random.sample(range(len(sample_group_files)), min(n_train, len(sample_group_files))):
        sample_img_path = os.path.join(sample_group_dir, sample_group_files[idx])
        if sample_img_path.endswith('JPG') or sample_img_path.endswith('png'):
          self.samples.append((sample_img_path, label))
    
    self.index_map = list(range(len(self.samples)))
    random.shuffle(self.index_map)

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    (img_path, label) = self.samples[self.index_map[idx]]
    img = Image.open(open(img_path, "rb"))
    img = self.transform(img)
    return (img, label, img_path[69:])
  
  @staticmethod
  def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.stack([torch.tensor(item[1]).long() for item in batch])
    img_paths = [item[2] for item in batch]
    return (imgs, labels, img_paths)

def leaves_loader(data_root, data_dir, n_train, batch_size, train_percentage):
  dataset = LeavesDataset(data_root, data_dir, n_train)
  num_train = int(len(dataset) * train_percentage)
  num_test = len(dataset) - num_train
  (train_dataset, test_dataset) = torch.utils.data.random_split(dataset, [num_train, num_test])
  train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=LeavesDataset.collate_fn, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=LeavesDataset.collate_fn, batch_size=batch_size, shuffle=True)
  return (train_loader, test_loader)

class LeavesNet(nn.Module):
  def __init__(self, sample_count, data_dir, caching):
    super(LeavesNet, self).__init__()

    # features for classification
    if data_dir == 'leaf_11':
      self.types = []
      self.margin = leaves_config.l11_margin
      self.shape = leaves_config.l11_shape
      self.texture = leaves_config.l11_texture
      self.venation = []
      self.labels = leaves_config.l11_labels
      self.dim = leaves_config.l11_dim
    elif data_dir == 'leaf_40':
      self.types = leaves_config.l40_type
      self.margin = leaves_config.l40_margin
      self.shape = leaves_config.l40_shape
      self.texture = leaves_config.l40_texture
      self.venation = leaves_config.l40_venation
      self.labels = leaves_config.l40_labels
      self.dim = leaves_config.l40_dim
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
    
    # Fully connected for 'type'
    self.type_fc = nn.Sequential(
      nn.Linear(self.dim, 1024),
      nn.ReLU(),
      nn.Linear(1024, len(self.types)),
      nn.Softmax(dim=1)
    )
    
    # Fully connected for 'margin'
    self.margin_fc = nn.Sequential(
      nn.Linear(self.dim, 1024),
      nn.ReLU(),
      nn.Linear(1024, len(self.margin)),
      nn.Softmax(dim=1)
    )

    # Fully connected for 'shape'
    self.shape_fc = nn.Sequential(
      nn.Linear(self.dim, 1024),
      nn.ReLU(),
      nn.Linear(1024, len(self.shape)),
      nn.Softmax(dim=1)
    )

    # Fully connected for 'texture'
    self.texture_fc = nn.Sequential(
      nn.Linear(self.dim, 1024),
      nn.ReLU(),
      nn.Linear(1024, len(self.texture)),
      nn.Softmax(dim=1)
    )

    # Fully connected for 'venation'
    self.venation_fc = nn.Sequential(
      nn.Linear(self.dim, 1024),
      nn.ReLU(),
      nn.Linear(1024, len(self.venation)),
      nn.Softmax(dim=1)
    )

    # Blackbox encoding identification chart
    if data_dir == 'leaf_11':
      self.bbox = blackbox.BlackBoxFunction(
                  leaves_config.classify_11,
                  (blackbox.DiscreteInputMapping(self.margin),
                  blackbox.DiscreteInputMapping(self.shape),
                  blackbox.DiscreteInputMapping(self.texture)),
                  blackbox.DiscreteOutputMapping(self.labels),
                  caching=caching,
                  sample_count=sample_count)
    elif data_dir == 'leaf_40':
      self.bbox = blackbox.BlackBoxFunction(
                  leaves_config.classify_40,
                  (blackbox.DiscreteInputMapping(self.types),
                  blackbox.DiscreteInputMapping(self.margin),
                  blackbox.DiscreteInputMapping(self.shape),
                  blackbox.DiscreteInputMapping(self.texture),
                  blackbox.DiscreteInputMapping(self.venation),),
                  blackbox.DiscreteOutputMapping(self.labels),
                  caching=caching,
                  sample_count=sample_count)
    else:
      raise Exception(f"Unknown directory: {data_dir}")

  def forward(self, x):
    x = self.cnn(x)
    # has_type = self.type_fc(x)
    has_margin = self.margin_fc(x)
    has_shape = self.shape_fc(x)
    has_texture = self.texture_fc(x)
    # has_venation = self.venation_fc(x)
    return (has_margin, has_shape, has_texture, 
            self.bbox(has_margin, has_shape, has_texture))

class Trainer():
  def __init__(self, train_loader, test_loader, learning_rate, sample_count, caching, gpu, save_model=False):
    if gpu >= 0:
      device = torch.device("cuda:%d" % gpu)
    else:
      device = torch.device("cpu")
    self.device = device
    self.network = LeavesNet(sample_count, caching) #.to(self.device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.save_model = save_model

    # Aggregated loss (initialized to be a huge number)
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
    for (i, (input, target, _)) in enumerate(iter):
      self.optimizer.zero_grad()
      input = input.to(self.device)
      target = target.to(self.device)
      (_, _, _, output) = self.network(input)
      loss = self.loss_fn(output, target)
      loss.backward()
      self.optimizer.step()
      total_correct += (output.argmax(dim=1)==target).float().sum()
      num_items += output.shape[0]
      correct_perc = 100. * total_correct / num_items
      train_loss += loss.item()
      avg_loss = train_loss / (i + 1)
      iter.set_description(f"[Train Epoch {epoch}] Loss: {avg_loss:.4f}, Overall Accuracy: {int(total_correct)}/{int(num_items)} ({correct_perc:.2f})%")

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = 0
    num_correct = 0
    test_loss = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for i, (input, target, img_paths) in enumerate(iter):
        input = input.to(self.device)
        target = target.to(self.device)
        (m, s, t, output) = self.network(input)
        num_items += output.shape[0]
        test_loss += self.loss_fn(output, target).item()
        num_correct += (output.argmax(dim=1)==target).float().sum()
        perc = 100.*num_correct/num_items
        avg_loss = test_loss / (i + 1)
        iter.set_description(f"[Test Epoch {epoch}] Avg loss: {avg_loss:.4f}, {int(num_correct)}/{int(num_items)} ({perc:.2f})%")
        # debug
        if epoch > 40:
          for i in range(len(img_paths)):
            dict = {}
            if output[i].argmax() != target[i]:  
              dict['epoch'] = epoch  
              dict['path'] = img_paths[i]
              dict['margin'] = leaves_config.l11_margin[m[i].argmax()]
              dict['shape'] = leaves_config.l11_shape[s[i].argmax()]
              dict['texture'] = leaves_config.l11_texture[t[i].argmax()]
              dict['prediction'] = leaves_config.l11_labels[output[i].argmax()]     
              with open('demo/leaf/leaf_wrong.csv', 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['epoch', 'path', 'margin', 'shape', 'texture', 'prediction'])
                writer.writerow(dict)
                csvfile.close()
    
    # Save the model
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
  # Argument parser
  parser = ArgumentParser("leaves")
  parser.add_argument("--model-name", type=str, default="leaves.pkl")
  parser.add_argument("--n-epochs", type=int, default=50)
  parser.add_argument("--sample-count", type=int, default=100)
  parser.add_argument("--gpu", type=int, default=-1)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--caching", type=bool, default=False)
  parser.add_argument("--cuda", action="store_true")
  args = parser.parse_args()

  random_seeds = [1234, 3177, 5848, 9175]
  train_nums = [50, 80, 75, 100, 150]
  train_percentages = [0.2, 0.25, 0.4, 0.5, 0.7]
  data_dirs = ['leaf_11', 'leaf_40']
  accuracies = ["accuracy epoch " + str(i+1) for i in range(args.n_epochs)]
  times = ["time epoch " + str(i+1) for i in range(args.n_epochs)]
  field_names = ['random seed', 'data_dir', 'num train'] + accuracies + times

  with open('demo/leaf/leaf_bbox.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()
    csvfile.close()
  
  with open('demo/leaf/leaf_wrong.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['epoch', 'path', 'margin', 'shape', 'texture', 'prediction'])
    writer.writeheader()
    csvfile.close()

  for data_dir in data_dirs:
    for i in range(len(train_nums)): # 10, 20, 30, 50, 100
      for seed in random_seeds:
        # Setup parameters
        torch.manual_seed(seed)
        random.seed(seed)

        # Load data
        data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../benchmarks/data"))
        model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/leaves"))
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        
        (train_loader, test_loader) = leaves_loader(data_root, data_dir, train_nums[i], args.batch_size, train_percentages[i])
        trainer = Trainer(train_loader, test_loader, args.learning_rate, args.sample_count, data_dir, args.caching, args.gpu)

        # Run
        dict = trainer.train(args.n_epochs)
        dict["random seed"] = seed
        dict['data_dir'] = data_dir
        dict["num train"] = int(train_percentages[i]*train_nums[i])
        with open('demo/leaf/leaf_bbox.csv', 'a', newline='') as csvfile:
          writer = csv.DictWriter(csvfile, fieldnames=field_names)
          writer.writerow(dict)
          csvfile.close()