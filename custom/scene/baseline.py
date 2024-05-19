import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from PIL import Image

from typing import *
import os
import random
import time
from argparse import ArgumentParser

from dataset import scenes

img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class SceneDataset(torch.utils.data.Dataset):
  """
  :param data_root, the root directory of the data folder
  """
  def __init__(
    self,
    data_root: str,
    data_dir: str,
    transform: Optional[Callable] = img_transform,
  ):
    self.transform = transform
    self.labels = scenes
    self.transform = transform
    
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
      for idx in range(len(sample_group_files)):
        sample_img_path = os.path.join(sample_group_dir, sample_group_files[idx])
        if sample_img_path.endswith('jpg'):
          self.samples.append((sample_img_path, sample_group_files[idx], label))
    
    self.index_map = list(range(len(self.samples)))
    random.shuffle(self.index_map)

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    (img_path, file_name, label) = self.samples[self.index_map[idx]]
    img = Image.open(open(img_path, "rb"))
    img = self.transform(img)
    return (img, file_name, label)
  
  @staticmethod
  def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    names = [item[1] for item in batch]
    labels = torch.stack([torch.tensor(item[2]).long() for item in batch])
    return (imgs, names, labels)

def scene_loader(data_root, batch_size):
  train_dataset = SceneDataset(data_root, "train")
  test_dataset = SceneDataset(data_root, "test")
  train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=SceneDataset.collate_fn, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=SceneDataset.collate_fn, batch_size=batch_size, shuffle=True)
  return (train_loader, test_loader)

class SceneNet(nn.Module):
  def __init__(self):
    super(SceneNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, 3)
    self.conv2 = nn.Conv2d(32, 32, 3)
    self.conv3 = nn.Conv2d(32, 64, 3)
    self.conv4 = nn.Conv2d(64, 64, 3)
    self.conv5 = nn.Conv2d(64, 128, 3)
    self.conv6 = nn.Conv2d(128, 128, 3)
    self.conv7 = nn.Conv2d(128, 256, 3)
    self.conv8 = nn.Conv2d(256, 256, 3)
    self.linear1 = nn.Linear(512, 512)
    self.linear2 = nn.Linear(512, 9)
  
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(F.relu(self.conv2(x)), 3)
    x = F.relu(self.conv3(x))
    x = F.max_pool2d(F.relu(self.conv4(x)), 3)
    x = F.relu(self.conv5(x))
    x = F.max_pool2d(F.relu(self.conv6(x)), 3)
    x = F.relu(self.conv7(x))
    x = F.max_pool2d(F.relu(self.conv8(x)), 3)
    x = x.view(-1, 512)
    x = F.relu(self.linear1(x))
    x = F.softmax(self.linear2(x), dim=1)
    return x

class Trainer():
  def __init__(self, model, train_loader, test_loader, learning_rate):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.network = model.to(self.device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = None

  def loss_fn(self, output, ground_truth):
    dim = output.shape[1]
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth]).to(self.device)
    return F.binary_cross_entropy(output, gt)
  
  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    total_correct = 0
    train_loss = 0
    for (input, _, target) in self.train_loader:
      self.optimizer.zero_grad()
      input = input.to(self.device)
      target = target.to(self.device)
      output = self.network(input)
      loss = self.loss_fn(output, target)
      loss.backward()
      self.optimizer.step()
      train_loss += loss.item()
      total_correct += (output.argmax(dim=1)==target).float().sum()
      num_items += output.shape[0]
      correct_perc = 100. * total_correct / num_items
    print(f"[Train Epoch {epoch}] Overall Accuracy: {correct_perc:.2f}%")
    return train_loss

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    num_correct = 0
    test_loss = 0
    with torch.no_grad():
      for (input, _, target) in self.test_loader:
        input = input.to(self.device)
        target = target.to(self.device)
        output = self.network(input)
        test_loss += self.loss_fn(output, target).item()
        num_correct += (output.argmax(dim=1)==target).float().sum()
        perc = 100.*num_correct/num_items
      print(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {int(num_correct)}/{int(num_items)} ({perc:.2f})%")
    
    if self.best_loss is None or test_loss < self.best_loss:
      self.best_loss = test_loss
      torch.save(self.network.state_dict(), model_dir+f"/baseline_best.pth")
    
    return float(num_correct/num_items)

  def train(self, n_epochs):
    for epoch in range(1, n_epochs+1):
      t0 = time.time()
      train_loss = self.train_epoch(epoch)
      t1 = time.time()
      acc = self.test_epoch(epoch)

if __name__ == "__main__":
  parser = ArgumentParser("scene")
  parser.add_argument("--model-name", type=str, default="scene.pkl")
  parser.add_argument("--n-epochs", type=int, default=50)
  parser.add_argument('--seed', default=1234, type=int)  
  parser.add_argument("--gpu", type=int, default=-1)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=1e-4)
  parser.add_argument("--cuda", action="store_true")
  args = parser.parse_args()

  torch.manual_seed(args.seed)
  random.seed(args.seed)

  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data/scene"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/scene"))
  if not os.path.exists(model_dir): os.makedirs(model_dir)
    
  model = SceneNet()
            
  (train_loader, test_loader) = scene_loader(data_root, args.batch_size)
  trainer = Trainer(model, train_loader, test_loader, args.learning_rate)

  trainer.train(args.n_epochs)