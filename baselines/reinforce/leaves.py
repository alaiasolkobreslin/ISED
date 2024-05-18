import os
import random
from typing import *
from PIL import Image
import csv
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

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

def leaves_loader(data_root, data_dir, batch_size, n_train, n_test):
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
  def __init__(self, dim):
    super(LeavesNet, self).__init__()
    self.net1 = LeafNet(6, 2304)
    self.net2 = LeafNet(5, 2304)
    self.net3 = LeafNet(4, 2304)
    self.dim = dim

  def forward(self, x):
    has_f1 = self.net1(x)
    has_f2 =  self.net2(x)
    has_f3 = self.net3(x)
    return has_f1, has_f2, has_f3
  
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
  
def loss_fn(data, target):
    pred = []
    x = data.flatten(0, -2).int()
    for margin, shape, texture in x:
      y_pred = classify_11(l11_margin[margin], l11_shape[shape], l11_texture[texture])
      pred.append(torch.tensor(l11_labels.index(y_pred)))
    acc = torch.where(torch.stack(pred).reshape(data.shape[:-1]) == target, 1., 0.)
    return acc

class Trainer():
  def __init__(self, model, loss_fn, train_loader, test_loader, model_dir, learning_rate, grad_type, dim, sample_count, seed):
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
    self.seed = seed
    self.cats = [6,5,4]

  def indecater_multiplier(self, batch_size):
    ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
    icr_mult = torch.zeros((self.dim, 6, self.sample_count, batch_size, self.dim))
    icr_replacement = torch.zeros((self.dim, 6, self.sample_count, batch_size, self.dim))
    for i in range(self.dim):
      for j in range(self.cats[i]):
        icr_mult[i,j,:,:,i] = 1
        icr_replacement[i,j,:,:,i] = j
    icr_mult = icr_mult.reshape((18, self.sample_count, batch_size, self.dim))[ind]
    icr_replacement = icr_replacement.reshape((18, self.sample_count, batch_size, self.dim))[ind]

    return icr_mult, icr_replacement

  def reinforce_grads(self, data, target):
    logits1, logits2, logits3 = self.network(data)
    d1 = torch.distributions.Categorical(logits=logits1)
    d2 = torch.distributions.Categorical(logits=logits2)
    d3 = torch.distributions.Categorical(logits=logits3)
    samples1 = d1.sample((self.sample_count,))
    samples2 = d2.sample((self.sample_count,))
    samples3 = d3.sample((self.sample_count,))
    samples = torch.stack((samples1, samples2, samples3), dim=2)
    f_sample = self.loss_fn(samples, target.unsqueeze(0))
    log_p_sample = d1.log_prob(samples1) + d2.log_prob(samples2) + d3.log_prob(samples3)
    f_mean = f_sample.mean(dim=0)

    reinforce = (f_sample.detach() * log_p_sample).mean(dim=0)
    reinforce_prob = (f_mean - reinforce).detach() + reinforce
    loss = -torch.log(reinforce_prob + 1e-8)
    loss = loss.mean(dim=0)
    return loss
  
  def indecater_grads(self, data, target):
    logits1, logits2, logits3 = self.network(data)
    d1 = torch.distributions.Categorical(logits=logits1)
    d2 = torch.distributions.Categorical(logits=logits2)
    d3 = torch.distributions.Categorical(logits=logits3)
    samples1 = d1.sample((self.sample_count,))
    samples2 = d2.sample((self.sample_count,))
    samples3 = d3.sample((self.sample_count,))
    samples = torch.stack((samples1, samples2, samples3), dim=2)
    f_sample = self.loss_fn(samples, target.unsqueeze(0))
    f_mean = f_sample.mean(dim=0)
    batch_size = data.shape[0]

    outer_samples = torch.stack([samples] * 15, dim=0)
    m, r = self.indecater_multiplier(batch_size)
    outer_samples = outer_samples * (1 - m) + r
    outer_loss = self.loss_fn(outer_samples, target.unsqueeze(0).unsqueeze(0).unsqueeze(0))
    
    variable_loss = outer_loss.mean(dim=2).permute(2,0,1)
    probs = torch.cat((F.softmax(logits1, dim=-1), F.softmax(logits2, dim=-1), F.softmax(logits3, dim=-1)), dim=1)
    indecater_expression = variable_loss.detach() * probs.unsqueeze(1)
    indecater_expression = indecater_expression.sum(dim=-1)
    indecater_expression = indecater_expression.sum(dim=-1)

    icr_prob = (f_mean - indecater_expression).detach() + indecater_expression
    loss = -torch.log(indecater_expression + 1e-8) # -torch.log(icr_prob + 1e-8)
    loss = loss.mean(dim=0)
    return loss
  
  def advanced_indecater_grads(self, data, target):
    logits1, logits2, logits3 = self.network(data)
    d1 = torch.distributions.Categorical(logits=logits1)
    d2 = torch.distributions.Categorical(logits=logits2)
    d3 = torch.distributions.Categorical(logits=logits3)
    samples1 = d1.sample((self.sample_count * self.dim,))
    samples2 = d2.sample((self.sample_count * self.dim,))
    samples3 = d3.sample((self.sample_count * self.dim,))
    samples = torch.stack((samples1, samples2, samples3), dim=2)
    f_sample = self.loss_fn(samples, target.unsqueeze(0))
    f_mean = f_sample.mean(dim=0)
    batch_size = data.shape[0]

    samples = samples.reshape((self.dim, self.sample_count, batch_size, self.dim))
    m, r = self.indecater_multiplier(batch_size)
    outer_samples = outer_samples * (1 - m) + r
    outer_loss = self.loss_fn(outer_samples, target.unsqueeze(0).unsqueeze(0).unsqueeze(0))
    
    variable_loss = outer_loss.mean(dim=2).permute(2,0,1)
    probs = torch.cat((F.softmax(logits1, dim=-1), F.softmax(logits2, dim=-1), F.softmax(logits3, dim=-1)), dim=1)
    indecater_expression = variable_loss.detach() * probs
    indecater_expression = indecater_expression.sum(dim=-1)
    indecater_expression = indecater_expression.sum(dim=-1)

    icr_prob = (f_mean - indecater_expression).detach() + indecater_expression
    loss = -torch.log(indecater_expression + 1e-8) # -torch.log(icr_prob + 1e-8)
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
        logits1, logits2, logits3 = self.network(data)
        logits = torch.stack((logits1.argmax(dim=-1), logits2.argmax(dim=-1), logits3.argmax(dim=-1)), dim=-1)
        pred = loss_fn(logits, target)
        correct += pred.sum()
      
    perc = float(correct/num_items)
    if self.best_loss is None or self.best_loss < perc:
      self.best_loss = perc
      torch.save(self.network.state_dict(), model_dir+f"/{self.grad_type}_{self.seed}_best.pth")

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

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("leaves")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--sample-count", type=int, default=7)
  parser.add_argument("--train-num", type=int, default=30)
  parser.add_argument("--test-num", type=int, default=10)
  parser.add_argument("--grad_type", type=str, default='reinforce')
  parser.add_argument("--data-dir", type=str, default="leaf_11")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--jit", action="store_true")
  parser.add_argument("--dispatch", type=str, default="parallel")
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  sample_count = args.sample_count
  grad_type = args.grad_type
  dim = 3

  accuracies = ["A " + str(i+1) for i in range(args.n_epochs)]
  times = ["T " + str(i+1) for i in range(args.n_epochs)]
  losses = ["L " + str(i+1) for i in range(args.n_epochs)]
  field_names = ['random seed', 'grad_type', 'task_type', 'sample count'] + accuracies + times + losses

  for seed in [548, 6787, 8371]:
      torch.manual_seed(seed)
      random.seed(seed)
      if grad_type == 'reinforce': sample_count = 100
      print(sample_count)
      print(seed)
      print(grad_type)

      # Data
      data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
      model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/leaves"))
      os.makedirs(model_dir, exist_ok=True)

      # Dataloaders
      train_loader, test_loader = leaves_loader(data_dir, args.data_dir, batch_size, args.train_num, args.test_num)

      # Create trainer and train
      trainer = Trainer(LeavesNet, loss_fn, train_loader, test_loader, model_dir, learning_rate, grad_type, dim, sample_count, seed)
      dict = trainer.train(n_epochs)
      dict["random seed"] = seed
      dict['grad_type'] = grad_type
      dict['task_type'] = "leaf"
      dict['sample count'] = sample_count
      with open('baselines/reinforce/icr.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writerow(dict)
        csvfile.close()