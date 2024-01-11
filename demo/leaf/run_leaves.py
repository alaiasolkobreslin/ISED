from typing import Optional, Callable, Tuple
import os
import random

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
    n_group = 0
    data_dir = os.path.join(data_root, data_dir)
    for sample_group in os.listdir(data_dir):
      sample_group_dir = os.path.join(data_dir, sample_group)
      if not os.path.isdir(sample_group_dir):
        continue
      sample_group_files = os.listdir(sample_group_dir)
      for idx in random.sample(range(len(sample_group_files)), min(n_train, len(sample_group_files))):
        sample_img_path = os.path.join(sample_group_dir, sample_group_files[idx])
        if sample_img_path.endswith('JPG') or sample_img_path.endswith('png'):
          self.samples.append((sample_img_path, n_group))
      n_group += 1
    
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
  def __init__(self, sample_count):
    super(LeavesNet, self).__init__()

    # features for classification
    self.margin = leaves_config.l11_margin
    self.shape = leaves_config.l11_shape
    self.texture = leaves_config.l11_texture
    self.labels = leaves_config.l11_labels
    self.dim = leaves_config.l11_dim
  
    # CNN
    self.cnn = nn.Sequential(
      nn.Conv2d(3, 32, 3, 1),
      nn.ReLU(),
      nn.MaxPool2d(3),
      nn.Conv2d(32, 64, 3, 1),
      nn.ReLU(),
      nn.MaxPool2d(3),
      nn.Flatten(),
    )
    
    # Fully connected for 'margin'
    self.margin_fc = nn.Sequential(
      nn.Linear(self.dim, 64),
      nn.ReLU(),
      nn.Linear(64, len(self.margin)),
      nn.Softmax(dim=1)
    )

    # Fully connected for 'shape'
    self.shape_fc = nn.Sequential(
      nn.Linear(self.dim, 64),
      nn.ReLU(),
      nn.Linear(64, len(self.shape)),
      nn.Softmax(dim=1)
    )

    # Fully connected for 'apex'
    self.texture_fc = nn.Sequential(
      nn.Linear(self.dim, 64),
      nn.ReLU(),
      nn.Linear(64, len(self.texture)),
      nn.Softmax(dim=1)
    )

    # Blackbox encoding identification chart
    self.bbox = blackbox.BlackBoxFunction(
      leaves_config.classify_11,
      (blackbox.DiscreteInputMapping(self.margin),
       blackbox.DiscreteInputMapping(self.shape),
       blackbox.DiscreteInputMapping(self.texture)),
      blackbox.DiscreteOutputMapping(self.labels),
      sample_count=sample_count)

  def forward(self, x):
    x = self.cnn(x)
    has_margin = self.margin_fc(x)
    has_shape = self.shape_fc(x)
    has_texture = self.texture_fc(x)
    return self.bbox(has_margin, has_shape, has_texture)

class Trainer():
  def __init__(self, train_loader, test_loader, learning_rate, sample_count, gpu, save_model=False):
    if gpu >= 0:
      device = torch.device("cuda:%d" % gpu)
    else:
      device = torch.device("cpu")
    self.device = device
    self.network = LeavesNet(sample_count) #.to(self.device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.save_model = save_model
    self.sample_count = sample_count

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
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for (input, target) in iter:
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
      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}, Overall Accuracy: {correct_perc:.4f}%")

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    num_correct = 0
    test_loss = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (input, target) in iter:
        input = input.to(self.device)
        target = target.to(self.device)
        output = self.network(input)
        test_loss += self.loss_fn(output, target).item()
        num_correct += (output.argmax(dim=1)==target).float().sum()
        perc = 100.*num_correct/num_items
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {num_correct}/{num_items} ({perc:.2f})%")
    
    # Save the model
    # if self.save_model and test_loss < self.min_test_loss:
    #  self.min_test_loss = test_loss
    #  torch.save(self.network, "../model/leaves/leaves_net.pkl")

  def train(self, n_epochs):
    for epoch in range(1, n_epochs+1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("leaves")
  parser.add_argument("--model-name", type=str, default="leaves.pkl")
  parser.add_argument("--n-epochs", type=int, default=50)
  parser.add_argument("--sample-count", type=int, default=100)
  parser.add_argument("--gpu", type=int, default=-1)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--train-percentage", type=float, default=0.8)
  parser.add_argument("--n-train", type=int, default=80)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--data-dir", type=str, default="leaf_11")
  parser.add_argument("--cuda", action="store_true")
  args = parser.parse_args()

  # Setup parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Load data
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../benchmarks/data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/leaves"))
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  (train_loader, test_loader) = leaves_loader(data_root, args.data_dir, args.n_train, args.batch_size, args.train_percentage)
  trainer = Trainer(train_loader, test_loader, args.learning_rate, args.sample_count, args.gpu)

  # Run
  trainer.train(args.n_epochs)



        