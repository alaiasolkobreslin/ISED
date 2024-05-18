import os
import random
from typing import *
from PIL import Image
import csv
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from openai import OpenAI
import json
from argparse import ArgumentParser

client = OpenAI(
  api_key='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'
)

queries = {}
dict_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../../finite_diff"))
with open(dict_dir+'/neuro-symbolic/leaf11.pkl', 'rb') as f: 
  queries = pickle.load(f)

l11_4_system = "You are an expert in classifying plant species based on the margin, shape, and texture of the leaves. You are designed to output a single JSON."
l11_4_one = ['entire', 'lobed', 'serrate']
l11_4_two = ['cordate', 'lanceolate', 'oblong', 'oval', 'ovate', 'palmate']
l11_4_three = ['glossy', 'papery', 'smooth']
l11_labels = ['Alstonia Scholaris', 'Citrus limon', 'Jatropha curcas', 'Mangifera indica', 'Ocimum basilicum',
              'Platanus orientalis', 'Pongamia Pinnata', 'Psidium guajava', 'Punica granatum', 'Syzygium cumini', 'Terminalia Arjuna']

def call_llm(plants, features):
  user_list = "* " + "\n* ".join(plants)
  question = "\n\nClassify each into one of: " + ", ".join(features) + "."
  format = "\n\nGive your answer without explanation."
  user_msg = user_list + question
  if user_msg in queries.keys():
    return queries[user_msg]
  raise Exception("WRONG")
  response = client.chat.completions.create(
              model="gpt-4-1106-preview", #
              messages=[
                {"role": "system", "content": l11_4_system},
                {"role": "user", "content": user_msg + format}
              ],
              top_p=0.00000001
            )
  if response.choices[0].finish_reason == 'stop':
    ans = response.choices[0].message.content
    if ans[3:7] == 'json': ans  = ans[7:-3]
    print(ans) #
    queries[user_msg] = ans
    return ans
  raise Exception("LLM failed to provide an answer") 

def parse_response(result, target):
  dict = json.loads(result)
  plants = []
  for plant in dict.keys():
    if dict[plant] == target: plants.append(plant)
  return plants

def classify_llm(feature1, feature2, feature3):
  result1 = call_llm(l11_labels, l11_4_one)
  plants1 = parse_response(result1, feature1)
  if len(plants1) == 1: return plants1[0]
  elif len(plants1) == 0: 
    plants1 = l11_labels # return 'unknown'
  else:
    results2 = call_llm(plants1, l11_4_two)
    plants2 = parse_response(results2, feature2)
    if len(plants2) == 1: return plants2[0]
    elif len(plants2) == 0: 
      plants2 = plants1  # return 'unknown'
    results3 = call_llm(plants2, l11_4_three)
    plants3 = parse_response(results3, feature3)
    if len(plants3) == 1: return plants3[0]
    elif len(plants3) == 0: return plants2[random.randrange(len(plants2))] # return 'unknown'
    else: return plants3[random.randrange(len(plants3))]

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
    self.labels = l11_labels
    
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
    self.f1 = l11_4_one
    self.f2 = l11_4_two
    self.f3 = l11_4_three

    self.net1 = LeafNet(len(self.f1), 2304)
    self.net2 = LeafNet(len(self.f2), 2304)
    self.net3 = LeafNet(len(self.f3), 2304)
    self.dim = dim

  def forward(self, x):
    has_f1 = self.net1(x)
    has_f2 =  self.net2(x)
    has_f3 = self.net3(x)
    return has_f1, has_f2, has_f3
  
def loss_fn(data, target):
    pred = []
    x = data.flatten(0, -2).int()
    for margin, shape, texture in x:
      r = (l11_4_one[margin], l11_4_two[shape], l11_4_three[texture])
      if r in cache: 
        y_pred = cache[r]
      else: 
        y_pred = classify_llm(*r)
        cache[r] = y_pred
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
    self.cats = [3,6,3]

  def indecater_multiplier(self, batch_size):
    ind = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14]
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

    outer_samples = torch.stack([samples] * 12, dim=0)
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
  
  def grads(self, data, target):
    if self.grad_type == 'reinforce':
      return self.reinforce_grads(data, target)
    elif self.grad_type == 'icr':
      return self.indecater_grads(data, target)

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
  parser.add_argument("--n-epochs", type=int, default=50)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--sample-count", type=int, default=9)
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
  grad_type = args.grad_type
  sample_count = args.sample_count
  dim = 3
  cache = {}

  accuracies = ["A " + str(i+1) for i in range(args.n_epochs)]
  times = ["T " + str(i+1) for i in range(args.n_epochs)]
  losses = ["L " + str(i+1) for i in range(args.n_epochs)]
  field_names = ['random seed', 'grad_type', 'task_type', 'sample count'] + accuracies + times + losses

  for seed in [3177, 5848, 9175, 8725, 1234]: # 1357, 2468, 548, 6787, 8371
      torch.manual_seed(seed)
      random.seed(seed)
      
      if grad_type == 'reinforce': sample_count = 100
      print(sample_count)
      print(seed)
      print(grad_type)

      # Data
      data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
      model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/leaves_llm"))
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