import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision

import math
import os
import json
import numpy as np
from time import time
import random
from typing import Optional, Callable
from PIL import Image

from argparse import ArgumentParser

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
    self.labels = ['Alstonia Scholaris', 'Citrus limon', 'Jatropha curcas', 'Mangifera indica', 'Ocimum basilicum',
                   'Platanus orientalis', 'Pongamia Pinnata', 'Psidium guajava', 'Punica granatum', 'Syzygium cumini', 'Terminalia Arjuna']
    
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
    return (img, label)
  
  @staticmethod
  def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.stack([torch.tensor(item[1]).long() for item in batch])
    return (imgs, labels)

def leaves_loader(data_root, data_dir, n_train, batch_size, n_test):
  num_class = 11
  dataset = LeavesDataset(data_root, data_dir, (n_train+n_test))
  num_train = n_train*num_class
  num_test = len(dataset) - num_train
  (train_dataset, test_dataset) = torch.utils.data.random_split(dataset, [num_train, num_test])
  train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=LeavesDataset.collate_fn, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=LeavesDataset.collate_fn, batch_size=batch_size, shuffle=True)
  return (train_loader, test_loader)

class LeafNet(nn.Module):
  def __init__(self, num_features, dim):
    super(LeafNet, self).__init__()
    self.num_features = num_features
    self.dim = dim

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

    # Fully connected for 'features'
    self.features_fc = nn.Sequential(
      nn.Linear(self.dim, self.dim),
      nn.ReLU(),
      nn.Linear(self.dim, self.num_features),
      nn.Softmax(dim=1)
    )
    
  def forward(self, x):
    x = self.cnn(x)
    x = self.features_fc(x)   
    return x  
  
class LeavesNet(nn.Module):
  def __init__(self):
    super(LeavesNet, self).__init__()
    self.net1 = LeafNet(6, 2304)
    self.net2 = LeafNet(5, 2304)
    self.net3 = LeafNet(4, 2304)

  def forward(self, x):
    has_f1 = self.net1(x)
    has_f2 = self.net2(x)
    has_f3 = self.net3(x)
    return (has_f1, has_f2, has_f3)

class RLLeavesNet(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    self.saved_log_probs = []
    self.rewards = []
    self.perception = LeavesNet()

  def forward(self, x):
    return self.perception.forward(x)

def classify_11(margin, shape, texture):
  if margin == 'serrate': return 'Ocimum basilicum'
  elif margin == 'indented': return 'Jatropha curcas'
  elif margin == 'lobed': return 'Platanus orientalis'
  elif margin == 'serrulate': return "Citrus limon"
  elif margin == 'entire':
    if shape == 'ovate': return 'Pongamia Pinnata'
    elif shape == 'lanceolate': return 'Mangifera indica'
    elif shape == 'oblong': return 'Syzygium cumini'
    elif shape == 'obovate': return "Psidium guajava"
    else:
      if texture == 'leathery': return "Alstonia Scholaris"
      elif texture == 'rough': return "Terminalia Arjuna"
      elif texture == 'glossy': return "Citrus limon"
      else: return "Punica granatum"
  else:
    if shape == 'elliptical': return 'Terminalia Arjuna'
    elif shape == 'lanceolate': return "Mangifera indica"
    else: return 'Syzygium cumini'

l11_margin = ['entire', 'indented', 'lobed', 'serrate', 'serrulate', 'undulate']
l11_shape = ['elliptical', 'lanceolate', 'oblong', 'obovate', 'ovate']
l11_texture = ['glossy', 'leathery', 'medium', 'rough']
l11_labels = ['Alstonia Scholaris', 'Citrus limon', 'Jatropha curcas', 'Mangifera indica', 'Ocimum basilicum',
              'Platanus orientalis', 'Pongamia Pinnata', 'Psidium guajava', 'Punica granatum', 'Syzygium cumini', 'Terminalia Arjuna']

def compute_reward(prediction, ground_truth):
    if prediction == l11_labels[ground_truth]:
        reward = 1
    else:
        reward = 0
    return reward

def validation(f1, f2, f3):
  f1 = f1.argmax(dim=1)
  f2 = f2.argmax(dim=1)
  f3 = f3.argmax(dim=1)

  predictions = []
  for i in range(len(f1)):
    prediction = classify_11(l11_margin[f1[i]], l11_shape[f2[i]], l11_texture[f3[i]])
    predictions.append(torch.tensor(l11_labels.index(prediction)))
  
  return torch.stack(predictions)

def final_output(model,ground_truth, f1, f2, f3, args):
  d1 = torch.distributions.categorical.Categorical(f1)
  d2 = torch.distributions.categorical.Categorical(f2)
  d3 = torch.distributions.categorical.Categorical(f3)

  s1 = d1.sample()
  s2 = d2.sample()
  s3 = d3.sample()

  model.saved_log_probs = d1.log_prob(s1)+d2.log_prob(s2)+d3.log_prob(s3)

  predictions = []
  for i in range(len(s1)):
    prediction = classify_11(l11_margin[s1[i]], l11_shape[s2[i]], l11_texture[s3[i]])
    predictions.append(torch.tensor(l11_labels.index(prediction)))
    reward = compute_reward(prediction,ground_truth[i])
    model.rewards.append(reward)
  
  return torch.stack(predictions)

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.learning_rate
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    elif not args.disable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
    
class Trainer():
  def __init__(self, train_loader, test_loader, model, path, args):
    self.network = model
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.path = path
    self.args = args
    self.best_loss = None
    self.best_reward = None
    self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    self.criterion = nn.BCEWithLogitsLoss()
  
  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    train_loss = 0

    eps = np.finfo(np.float32).eps.item()
    for i, (images, target) in enumerate(self.train_loader):
      images = images.to(self.args.gpu_id)
      target = target.to(self.args.gpu_id)
      f1, f2, f3 = self.network(images)
      final_output(model,target,f1,f2,f3,args)
      rewards = np.array(model.rewards)
      rewards_mean = rewards.mean()
      rewards = (rewards - rewards.mean())/(rewards.std() + eps)
      policy_loss = torch.zeros(len(rewards), requires_grad=True)
      
      for n, (reward, log_prob) in enumerate(zip(rewards, model.saved_log_probs)):
        policy_loss[n].data += (-log_prob.cpu()*reward)
      self.optimizer.zero_grad()
      
      policy_loss = policy_loss.sum()

      #if args.clip_grad_norm > 0:
      #  nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=args.clip_grad_norm, norm_type=2)

      num_items += images.size(0)
      train_loss += float(policy_loss.item() * images.size(0))
      policy_loss.backward()

      self.optimizer.step()
      
      if args.print_freq >= 0 and i % args.print_freq == 0:
        avg_loss = train_loss/num_items
        print(f'[rl][Epoch {epoch}][Train][{i}/{len(train_loader)}] \t AvgLoss: {avg_loss:.4f} \t AvgRewards: {rewards_mean:.4f}')
        stats2 = {'epoch': epoch, 'train': i, 'avr_train_loss': avg_loss, 'avr_train_reward': rewards_mean}
        with open(f"demo/model/nasr/detail_log.txt", "a") as f:
          f.write(json.dumps(stats2) + "\n")  

      model.rewards = []
      model.shared_log_probs = []
      torch.cuda.empty_cache()  
    
    return (train_loss/num_items), rewards_mean

  def test_epoch(self, epoch, time_begin):
    self.network.eval()
    num_items = 0
    test_loss = 0
    rewards_value = 0
    num_correct = 0

    eps = np.finfo(np.float32).eps.item()
    with torch.no_grad():
      for i, (images, target) in enumerate(self.test_loader):
        images = images.to(self.args.gpu_id)
        target = target.to(self.args.gpu_id)
        
        f1, f2, f3 = self.network(images)
        output = final_output(model,target,f1,f2,f3,args)

        rewards = np.array(model.rewards)
        rewards_mean = rewards.mean()
        rewards_value += float(rewards_mean * images.size(0))
        rewards = (rewards - rewards.mean())/(rewards.std() + eps)

        policy_loss = []
        for reward, log_prob in zip(rewards, model.saved_log_probs):
          policy_loss.append(-log_prob*reward)
        policy_loss = (torch.stack(policy_loss)).sum()

        num_items += images.size(0)
        test_loss += float(policy_loss.item() * images.size(0))
        model.rewards = []
        model.saved_log_probs = []
        torch.cuda.empty_cache()

        # output = validation(f1, f2, f3)
        num_correct += (output==target.cpu()).sum()
        perc = 100.*num_correct/num_items
        
        if self.best_loss is None or test_loss < self.best_loss:
          self.best_loss = test_loss
          torch.save(self.network.state_dict(), f'{self.path}/checkpoint_best_L.pth')
    
    avg_loss = (test_loss / num_items)
    avg_reward = (rewards_value/num_items)  
    total_mins = (time() - time_begin) / 60
    print(f"[Test Epoch {epoch}] {int(num_correct)}/{int(num_items)} ({perc:.2f})%")
    print(f'----[rl][Epoch {epoch}] \t \t AvgLoss {avg_loss:.4f} \t \t AvgReward {avg_reward:.4f} \t \t Time: {total_mins:.2f} ')
    
    return avg_loss, rewards_mean

  def train(self, n_epochs):
    time_begin = time()
    with open(f"{self.path}/log.txt", 'w'): pass
    with open(f"{self.path}/detail_log.txt", 'w'): pass
    for epoch in range(1, n_epochs+1):
      lr = adjust_learning_rate(self.optimizer, epoch, self.args)
      train_loss = self.train_epoch(epoch)
      test_loss = self.test_epoch(epoch, time_begin)
      stats = {'epoch': epoch, 'lr': lr, 'train_loss': train_loss, 'val_loss': test_loss, 'best_loss': self.best_loss}
      with open(f"{self.path}/log.txt", "a") as f: 
        f.write(json.dumps(stats) + "\n")
    torch.save(self.network.state_dict(), f'{self.path}/checkpoint_last.pth')

if __name__ == "__main__":
  parser = ArgumentParser('leaf')
  parser.add_argument('--gpu-id', default='cuda:0', type=str)
  parser.add_argument('-j', '--workers', default=0, type=int)
  parser.add_argument('--print-freq', default=5, type=int)

  parser.add_argument('--epochs', default=100, type=int)
  parser.add_argument('--warmup', default=10, type=int)
  parser.add_argument('-b', '--batch-size', default=16, type=int)
  parser.add_argument('--learning-rate', default=0.0001, type=float)
  parser.add_argument('--weight-decay', default=3e-1, type=float)
  parser.add_argument('--clip-grad-norm', default=1., type=float)
  parser.add_argument('--disable-cos', action='store_true')

  args = parser.parse_args()

  train_nums = 30
  test_nums = 10
  data_dir = 'leaf_11'

  torch.manual_seed(1234)
  random.seed(1234)

  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
  model_dir = os.path.join('demo/model', 'nasr')
  os.makedirs(model_dir, exist_ok=True)

  model = RLLeavesNet()
  model.to(args.gpu_id)

  (train_loader, test_loader) = leaves_loader(data_root, data_dir, train_nums, args.batch_size, test_nums)
  trainer = Trainer(train_loader, test_loader, model, model_dir, args)
  trainer.train(args.epochs)