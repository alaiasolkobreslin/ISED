import os
import random
from typing import *
import csv
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser

from configs import classify_llm
from dataset import scene_loader, scenes, objects, SceneNet, prepare_inputs

def loss_fn(data, target, lens):
    pred = []
    dim = data.shape[:-1].numel()
    data = data.flatten(0,-2)
    for d in range(dim):
      ind = 0
      for n in range(len(lens)):
        i = data[d][ind:ind+lens[n]]
        ind += lens[n]
        input = [objects[int(j)] for j in i]
        input.sort()
        y_pred = classify_llm(input)
        pred.append(torch.tensor(scenes.index(y_pred)))
    acc = torch.where(torch.stack(pred).reshape(dim, -1) == target, 1., 0.)
    return acc

class Trainer():
  def __init__(self, loss_fn, train_loader, test_loader, model_dir, learning_rate, grad_type, sample_count, seed):
    self.model_dir = model_dir
    self.device = torch.device("cpu")
    self.network = SceneNet()
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = None
    self.grad_type = grad_type
    self.sample_count = sample_count
    self.loss_fn = loss_fn
    self.seed = seed
    self.dict = {}

  def reinforce_grads(self, data, cls_id, conf, target, lens):
    logits = self.network(data, cls_id, conf)
    d = torch.distributions.Categorical(logits=logits)
    samples = d.sample((self.sample_count,))
    f_sample = self.loss_fn(samples, target.unsqueeze(0), lens) # 16 * 9
    f_mean = f_sample.mean(dim=0)

    log_p_sample = d.log_prob(samples) #sum(dim=-1)
    log_p_sample = [log_p_sample[:,lens[:i].sum():lens[:i+1].sum()].sum(dim=-1) for i, _ in enumerate(lens)]
    log_p_sample = torch.stack(log_p_sample, dim=1)

    reinforce = (f_sample.detach() * log_p_sample).mean(dim=0)
    reinforce_prob = (f_mean - reinforce).detach() + reinforce
    loss = -torch.log(reinforce_prob + 1e-8)
    loss = loss.mean(dim=0)
    return loss
  
  def indecater_grads(self, data, cls_id, conf, target, lens):
    logits = self.network(data, cls_id, conf)
    d = torch.distributions.Categorical(logits=logits)

    outer_samples = d.sample((self.sample_count,))
    outer_loss = self.loss_fn(outer_samples, target.unsqueeze(0), lens)
    variable_loss = outer_loss.mean(dim=0).unsqueeze(-1).unsqueeze(-1)

    logits = [F.pad(F.softmax(logits[lens[:i].sum():lens[:i+1].sum(), :], dim=-1), (0,0,0,10-lens[i])) for i, _ in enumerate(lens)]
    logits = torch.stack(logits, dim=0)
    
    indecater_expression = variable_loss.detach() * logits
    indecater_expression = indecater_expression.sum(dim=-1)
    indecater_expression = indecater_expression.sum(dim=-1)

    loss = -torch.log(indecater_expression + 1e-8) # -torch.log(icr_prob + 1e-8)
    loss = loss.mean(dim=0)
    return loss
  
  def grads(self, data, cls, conf, target, box_len):
    if self.grad_type == 'reinforce':
      return self.reinforce_grads(data, cls, conf, target, box_len)
    elif self.grad_type == 'icr':
      return self.indecater_grads(data, cls, conf, target, box_len)

  def train_epoch(self, epoch):
    train_loss = 0
    self.network.train()
    for (input, file, target) in self.train_loader:
      self.optimizer.zero_grad()
      box_len, cls, input, conf = prepare_inputs(input, file, self.dict)
      target = target.to(self.device)
      loss = self.grads(input, cls, conf, target, box_len)
      train_loss += loss
      loss.backward()
      self.optimizer.step()
    print(f"[Epoch {epoch}] : {train_loss}")
    return train_loss

  def test(self):
    num_items = len(self.test_loader.dataset)
    correct = 0
    with torch.no_grad():
      for (input, file, target) in self.test_loader:
        box_len, cls, input, conf = prepare_inputs(input, file, self.dict)
        target = target.to(self.device)
        output = self.network(input.to(self.device), cls.to(self.device), conf.to(self.device))
        pred = self.loss_fn(output.argmax(dim=1).unsqueeze(0), target.unsqueeze(0), box_len)
        correct += pred.sum()
      
    perc = float(correct/num_items)
    if self.best_loss is None or self.best_loss < perc:
      self.best_loss = perc
      torch.save(self.network, model_dir+f"/{self.grad_type}_{self.seed}_best.pkl")

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
    parser = ArgumentParser("scene_reinforce")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--grad_type", type=str, default='reinforce')
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--dispatch", type=str, default="parallel")
    args = parser.parse_args()
    
    grad_type = args.grad_type

    # Parameters
    accuracies = ["A " + str(i+1) for i in range(args.n_epochs)]
    times = ["T " + str(i+1) for i in range(args.n_epochs)]
    losses = ["L " + str(i+1) for i in range(args.n_epochs)]
    field_names = ['random seed', 'sample_count', 'grad_type'] + accuracies + times + losses
    
    for seed in [3177, 5848, 9175, 8725, 1234, 1357, 2468, 548, 6787, 8371]:
        torch.manual_seed(seed)
        random.seed(seed)

        # Data
        data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data/scene"))
        model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/scene/"+grad_type))
        os.makedirs(model_dir, exist_ok=True)

        (train_loader, test_loader) = scene_loader(data_root, args.batch_size)
        trainer = Trainer(loss_fn, train_loader, test_loader, model_dir, args.learning_rate, grad_type, args.sample_count, seed)

        dict = trainer.train(args.n_epochs)
        dict["random seed"] = seed
        dict['grad_type'] = grad_type
        dict['sample_count'] = args.sample_count
        with open('scene/bbox.csv', 'a', newline='') as csvfile:
          writer = csv.DictWriter(csvfile, fieldnames=field_names)
          writer.writerow(dict)
          csvfile.close()