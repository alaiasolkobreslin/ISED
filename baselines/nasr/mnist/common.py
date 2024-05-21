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
import time

from argparse import ArgumentParser

import os
import random
from typing import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

def compute_reward(prediction, ground_truth):
    if prediction == ground_truth:
        reward = 1.0
    else:
        reward = 0.0
    return reward

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.learning_rate
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    elif not args.disable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def validate(val_loader, model, final_output, args):
    model.eval()
    loss_value = 0
    reward_value = 0
    n = 0
    eps = np.finfo(np.float32).eps.item()
    
    iter = tqdm(val_loader, total=len(val_loader))

    with torch.no_grad():
        for i, (images, target) in enumerate(iter):
            images = tuple(image.to(args.gpu_id) for image in images)
            target = target.to(args.gpu_id)

            preds = model(images)
            final_output(model,target,args, *preds) # this populates model.rewards
            rewards = np.array(model.rewards)
            rewards_mean = rewards.mean()
            reward_value += float(rewards_mean * images[0].size(0))
            rewards = (rewards - rewards.mean())/(rewards.std() + eps)
            policy_loss = []
            for reward, log_prob in zip(rewards, model.saved_log_probs):
                policy_loss.append(-log_prob*reward)
            policy_loss = (torch.stack(policy_loss)).sum()

            n += images[0].size(0)
            loss_value += float(policy_loss.item() * images[0].size(0))
            model.rewards = []
            model.saved_log_probs = []
            torch.cuda.empty_cache()

            iter.set_description(f"[Val][{i}] AvgLoss: {loss_value/n:.4f} AvgRewards: {rewards_mean:.4f}")
    
    avg_loss = (loss_value / n)

    return avg_loss, rewards_mean


class Trainer():
  def __init__(self, train_loader, valid_loader, test_loader, model, path, final_output, args):
    self.network = model
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    self.test_loader = test_loader
    self.model = model
    self.path = path
    self.final_output = final_output
    self.args = args
    self.best_loss = None
    self.best_reward = None
    self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    self.criterion = nn.BCEWithLogitsLoss()
  
  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    train_loss = 0
    
    iter = tqdm(self.train_loader, total=len(self.train_loader))

    eps = np.finfo(np.float32).eps.item()
    for i, (images, target) in enumerate(iter):
      images = tuple(image.to(self.args.gpu_id) for image in images)
      target = target.to(self.args.gpu_id)
      preds = self.network(images)
      self.final_output(self.model,target,self.args,*preds)
      rewards = np.array(self.model.rewards)
      rewards_mean = rewards.mean()
      rewards = (rewards - rewards.mean())/(rewards.std() + eps)
      policy_loss = torch.zeros(len(rewards), requires_grad=True)
      
      for n, (reward, log_prob) in enumerate(zip(rewards, self.model.saved_log_probs)):
        policy_loss[n].data += (-log_prob.cpu()*reward)
      self.optimizer.zero_grad()
      
      policy_loss = policy_loss.sum()

      num_items += images[0].size(0)
      train_loss += float(policy_loss.item() * images[0].size(0))
      policy_loss.backward()

      self.optimizer.step()
      
      avg_loss = train_loss/num_items
      iter.set_description(f"[Train Epoch {epoch}] AvgLoss: {avg_loss:.4f} AvgRewards: {rewards_mean:.4f}")
      
      if self.args.print_freq >= 0 and i % self.args.print_freq == 0:
        stats2 = {'epoch': epoch, 'train': i, 'avr_train_loss': avg_loss, 'avr_train_reward': rewards_mean}
        with open(f"model/nasr/detail_log.txt", "a") as f:
          f.write(json.dumps(stats2) + "\n")
      self.model.rewards = []
      self.model.shared_log_probs = []
      torch.cuda.empty_cache()
    
    return (train_loss/num_items), rewards_mean

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = 0
    test_loss = 0
    rewards_value = 0
    num_correct = 0
    
    iter = tqdm(self.test_loader, total=len(self.test_loader))

    eps = np.finfo(np.float32).eps.item()
    with torch.no_grad():
      for i, (images, target) in enumerate(iter):
        images = tuple(image.to(self.args.gpu_id) for image in images)
        target = target.to(self.args.gpu_id)
        
        preds = self.network(images)
        output = self.final_output(self.model,target,self.args,*preds)

        rewards = np.array(self.model.rewards)
        rewards_mean = rewards.mean()
        rewards_value += float(rewards_mean * images[0].size(0))
        rewards = (rewards - rewards.mean())/(rewards.std() + eps)

        policy_loss = []
        for reward, log_prob in zip(rewards, self.model.saved_log_probs):
          policy_loss.append(-log_prob*reward)
        policy_loss = (torch.stack(policy_loss)).sum()

        num_items += images[0].size(0)
        test_loss += float(policy_loss.item() * images[0].size(0))
        self.model.rewards = []
        self.model.saved_log_probs = []
        torch.cuda.empty_cache()

        num_correct += (output==target).sum()
        perc = 100.*num_correct/num_items
        
        iter.set_description(f"[Test Epoch {epoch}] {int(num_correct)}/{int(num_items)} ({perc:.2f})%")
        
        if self.best_loss is None or test_loss < self.best_loss:
          self.best_loss = test_loss
          torch.save(self.network.state_dict(), f'{self.path}/checkpoint_best_L.pth')
    
    avg_loss = (test_loss / num_items)
    
    return avg_loss, rewards_mean, perc

  def train(self, n_epochs):
    ckpt_path = os.path.join('outputs', 'nasr/')
    best_loss = None
    best_reward = None
    with open(f"{self.path}/log.txt", 'w'): pass
    with open(f"{self.path}/detail_log.txt", 'w'): pass
    for epoch in range(1, n_epochs+1):
        lr = adjust_learning_rate(self.optimizer, epoch, self.args)
        train_loss, train_rewards = self.train_epoch(epoch)
        loss, valid_rewards = validate(self.valid_loader, self.model, self.final_output, self.args)
        test_loss, _, test_accuracy = self.test_epoch(epoch)
      
        if best_reward is None or valid_rewards > best_reward :
            best_reward = valid_rewards
            torch.save(self.model.state_dict(), f'{ckpt_path}/checkpoint_best_R.pth')
        
        if best_loss is None or loss < best_loss :
            best_loss = loss
            torch.save(self.model.state_dict(), f'{ckpt_path}/checkpoint_best_L.pth')
      
        stats = {'epoch': epoch, 'lr': lr, 'train_loss': train_loss, 
                    'val_loss': loss, 'best_loss': best_loss , 
                    'train_rewards': train_rewards, 'valid_rewards': valid_rewards,
                    'test_loss': test_loss, 'test_accuracy': test_accuracy.item()}
        with open(f"{self.path}/log.txt", "a") as f:
            f.write(json.dumps(stats) + "\n")
    torch.save(self.network.state_dict(), f'{self.path}/checkpoint_last.pth')
    
    # Testing the best model
    self.model.load_state_dict(torch.load(f'{ckpt_path}/checkpoint_best_R.pth'))
    avg_loss, rewards_mean, perc = self.test_epoch(n_epochs+1)
    print(f"Best Model Test Loss: {avg_loss}")
    print(f"Best Model Test Reward: {rewards_mean}")
    print(f"Best Model Test Accuracy: {perc}")
