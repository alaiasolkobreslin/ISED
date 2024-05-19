import torch
import torch.nn as nn

import os
import numpy as np
from time import time
import random
from argparse import ArgumentParser

from dataset import scenes, scene_loader, objects, SceneNet, prepare_inputs
from configs import classify_llm
    
def compute_reward(prediction, ground_truth):
    if prediction == scenes[ground_truth]: reward = 1
    else: reward = 0
    return reward

class Trainer():
  def __init__(self, train_loader, test_loader, path, seed, args):
    self.device = torch.device('cpu')
    self.network = SceneNet().to(self.device)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.path = path
    self.args = args
    self.best_loss = None
    self.best_reward = None
    self.best_acc = None
    self.optimizer = torch.optim.Adam(self.network.parameters(), lr=args.learning_rate)
    self.criterion = nn.BCEWithLogitsLoss()
    self.seed = seed
    self.dict = {}
  
  def final_output(self, gt, x, lens):
    d = torch.distributions.categorical.Categorical(x)
    s = d.sample()
    self.network.saved_log_probs = d.log_prob(s)

    predictions = []
    for n in range(len(lens)):
      ind = 0
      i = s[ind:ind+lens[n]]
      ind += lens[n]
      input = [objects[int(j)] for j in i]
      input.sort()
      y_pred = classify_llm(input)
      predictions.append(torch.tensor(scenes.index(y_pred)))
      reward = compute_reward(y_pred, gt[n])
      self.network.rewards.append(reward)
      
    return torch.stack(predictions)
  
  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    train_loss = 0

    eps = np.finfo(np.float32).eps.item()
    for i, (input, file, target) in enumerate(self.train_loader):
      box_len, cls, input, conf = prepare_inputs(input, file, self.dict)
      target = target.to(self.device)
      output = self.network(input.to(self.device), cls.to(self.device), conf.to(self.device))
      self.final_output(target, output, box_len)
      rewards = np.array(self.network.rewards)
      rewards_mean = rewards.mean()
      rewards = (rewards - rewards.mean())/(rewards.std() + eps)
      policy_loss = torch.zeros(len(rewards), requires_grad=True)
      
      for n, (reward, log_prob) in enumerate(zip(rewards, self.network.saved_log_probs)):
        policy_loss[n].data += (-log_prob.cpu()*reward)
      
      self.optimizer.zero_grad()
      policy_loss = policy_loss.sum()
      nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=args.clip_grad_norm, norm_type=2)

      num_items += input.size(0)
      train_loss += float(policy_loss.item() * input.size(0))
      policy_loss.backward()

      self.optimizer.step()
      
      if args.print_freq >= 0 and i % args.print_freq == 0:
        avg_loss = train_loss/num_items
        print(f'[rl][Epoch {epoch}][Train][{i}/{len(train_loader)}] \t AvgLoss: {avg_loss:.4f} \t AvgRewards: {rewards_mean:.4f}')

      self.network.rewards = []
      self.network.saved_log_probs = []
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
      for input, file, target in self.test_loader:
        box_len, cls, input, conf = prepare_inputs(input, file, self.dict)
        target = target.to(self.device)
        output = self.network(input.to(self.device), cls.to(self.device), conf.to(self.device))
        output = self.final_output(target, output, box_len)

        rewards = np.array(self.network.rewards)
        rewards_mean = rewards.mean()
        rewards_value += float(rewards_mean * input.size(0))
        rewards = (rewards - rewards.mean())/(rewards.std() + eps)

        policy_loss = []
        for reward, log_prob in zip(rewards, self.network.saved_log_probs):
          policy_loss.append(-log_prob*reward)
        policy_loss = (torch.stack(policy_loss)).sum()

        num_items += input.size(0)
        test_loss += float(policy_loss.item() * input.size(0))
        self.network.rewards = []
        self.network.saved_log_probs = []
        torch.cuda.empty_cache()

        num_correct += (output==target.to(self.device)).sum()
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
  parser = ArgumentParser('scene-nasr')
  parser.add_argument('-j', '--workers', default=0, type=int)
  parser.add_argument('--print-freq', default=10, type=int)
  parser.add_argument('--n-epochs', default=50, type=int)
  parser.add_argument('--seed', default=1234, type=int)
  parser.add_argument('-b', '--batch-size', default=16, type=int)
  parser.add_argument('--learning-rate', default=5e-4, type=float)
  parser.add_argument('--warmup', default=10, type=int)
  parser.add_argument('--weight-decay', default=3e-1, type=float)
  parser.add_argument('--clip-grad-norm', default=1., type=float)
  args = parser.parse_args()

  torch.manual_seed(args.seed)
  random.seed(args.seed)

  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data/scene"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/scene/nasr"))
  os.makedirs(model_dir, exist_ok=True)

  (train_loader, test_loader) = scene_loader(data_root, args.batch_size)
  trainer = Trainer(train_loader, test_loader, model_dir, args.seed, args)
    
  trainer.train(args.n_epochs)
