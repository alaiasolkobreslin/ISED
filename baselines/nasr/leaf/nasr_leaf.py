import torch
import torch.nn as nn
import os
import numpy as np
from time import time
import random

from argparse import ArgumentParser
from dataset import leaves_loader, LeafNet

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
    
class Trainer():
  def __init__(self, train_loader, test_loader, model, path, seed, args):
    self.network = model
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.path = path
    self.args = args
    self.best_loss = None
    self.best_reward = None
    self.best_acc = None
    self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    self.criterion = nn.BCEWithLogitsLoss()
    self.seed = seed
  
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

      nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=args.clip_grad_norm, norm_type=2)

      num_items += images.size(0)
      train_loss += float(policy_loss.item() * images.size(0))
      policy_loss.backward()

      self.optimizer.step()
      
      if args.print_freq >= 0 and i % args.print_freq == 0:
        avg_loss = train_loss/num_items
        print(f'[rl][Epoch {epoch}][Train][{i}/{len(train_loader)}] \t AvgLoss: {avg_loss:.4f} \t AvgRewards: {rewards_mean:.4f}')

      model.rewards = []
      model.saved_log_probs = []
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
      acc = float(num_correct/num_items)
        
      if self.best_loss is None or test_loss < self.best_loss:
        self.best_loss = test_loss
      
      if self.best_reward is None or rewards_value > self.best_reward:
        self.best_reward = rewards_value
      
      if self.best_acc is None or acc > self.best_acc:
        self.best_acc = acc
      
    avg_loss = (test_loss / num_items)
    avg_reward = (rewards_value/num_items)  
    total_mins = (time() - time_begin) / 60
    print(f"[Test Epoch {epoch}] {int(num_correct)}/{int(num_items)} ({acc*100:.2f})%")
    print(f'----[rl][Epoch {epoch}] \t \t AvgLoss {avg_loss:.4f} \t \t AvgReward {avg_reward:.4f} \t \t Time: {total_mins:.2f} ')
    return avg_loss, avg_reward, acc

  def train(self, n_epochs):
    time_begin = time()
    for epoch in range(1, n_epochs+1):
      t0 = time()
      train_loss, train_reward = self.train_epoch(epoch)
      t1 = time()
      test_loss, test_reward, test_acc = self.test_epoch(epoch, time_begin)
      stats = {'epoch': epoch, 
               'train_loss': train_loss, 'val_loss': test_loss, 'best_loss': self.best_loss,
               'train_reward': train_reward, 'val_rewards': test_reward, 'best_reward': self.best_reward,
               'test_acc': test_acc, 'best_acc': self.best_acc}

if __name__ == "__main__":
  parser = ArgumentParser('leaf')
  parser.add_argument('--gpu-id', default=0, type=int)
  parser.add_argument('-j', '--workers', default=4, type=int)
  parser.add_argument('--print-freq', default=10, type=int)
  parser.add_argument('--block-len', default=3, type=int)
  parser.add_argument('--seed', default=1234, type=int) 

  parser.add_argument('--n-epochs', default=100, type=int)
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

  torch.manual_seed(args.seed)
  random.seed(args.seed)

  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../../data"))
  model_dir = 'nasr'
  os.makedirs(model_dir, exist_ok=True)

  model = RLLeavesNet()
  model.to(args.gpu_id)

  (train_loader, test_loader) = leaves_loader(data_root, data_dir, train_nums, args.batch_size, test_nums)
  trainer = Trainer(train_loader, test_loader, model, model_dir, args.seed, args)

  # Run
  trainer.train(args.n_epochs)