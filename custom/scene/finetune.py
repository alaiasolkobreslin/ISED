import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

import csv
import time
from argparse import ArgumentParser
import os
import random

from dataset import scene_loader, scenes, SceneNet, prepare_inputs, objects
import blackbox
from configs import classify_llm

class ISEDSceneNet(nn.Module):
    def __init__(self, sample_count, max_det):
        super(ISEDSceneNet, self).__init__()
        self.net = SceneNet()
        self.blackbox = blackbox.BlackBoxFunction(
                          classify_llm,
                          (blackbox.ListInputMapping(max_det, blackbox.DiscreteInputMapping(objects)),),
                          blackbox.DiscreteOutputMapping(scenes),
                          sample_count = sample_count,
                          aggregate_strategy = "addmult")
    
    def forward(self, x, pred, box_len, conf):
      x = self.net(x, pred, conf)
      x = self.blackbox(blackbox.ListInput(x, box_len))
      return x

class Trainer():
  def __init__(self, train_loader, test_loader, learning_rate, sample_count, max_det, model_dir, seed):
    self.device = torch.device("cpu")
    self.network = ISEDSceneNet(sample_count, max_det).to(self.device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = None
    self.model_dir = model_dir
    self.seed = seed
    self.dict = {}
  
  def loss_fn(self, output, ground_truth):
    dim = output.shape[1]
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth]).to(self.device)
    return F.binary_cross_entropy(output, gt)

  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    total_correct = 0
    train_loss = 0
    for (input, file, target) in self.train_loader:
      self.optimizer.zero_grad()
      box_len, cls, input, conf = prepare_inputs(input, file, self.dict)
      target = target.to(self.device)
      output = self.network(input.to(self.device), cls.to(self.device), box_len.to(self.device), conf.to(self.device))
      loss = self.loss_fn(output, target)
      loss.backward()
      self.optimizer.step()
      total_correct += (output.argmax(dim=1)==target).float().sum()
      num_items += output.shape[0]
      correct_perc = 100. * total_correct / num_items
      train_loss += loss.item()
    print(f"[Train Epoch {epoch}] Overall Accuracy: {correct_perc:.2f}%")
    return train_loss

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    num_correct = 0
    test_loss = 0
    with torch.no_grad():
      for (input, file, target) in self.test_loader:
        box_len, cls, input, conf = prepare_inputs(input, file, self.dict)
        target = target.to(self.device)
        output = self.network(input.to(self.device), cls.to(self.device), box_len.to(self.device), conf.to(self.device))
        test_loss += self.loss_fn(output, target).item()
        num_correct += (output.argmax(dim=1)==target).float().sum()
      perc = 100.*num_correct/num_items
      print(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {int(num_correct)}/{int(num_items)} ({perc:.2f})%")
    
    if self.best_loss is None or test_loss < self.best_loss:
      self.best_loss = test_loss
      torch.save(self.network.state_dict(), model_dir+f"/{self.seed}_best.pth")
    
    return float(num_correct/num_items)

  def train(self, n_epochs):
    dict = {}
    for epoch in range(1, n_epochs+1):
      t0 = time.time()      
      train_loss = self.train_epoch(epoch)
      t1 = time.time()
      acc = self.test_epoch(epoch)
      dict["L " + str(epoch)] = round(float(train_loss), ndigits=6)
      dict["A " + str(epoch)] = round(float(acc), ndigits=6)
      dict["T " + str(epoch)] = round(t1 - t0, ndigits=6)
    torch.save(self.network.state_dict(), model_dir+f"/{self.seed}_last.pth")
    return dict

if __name__ == "__main__":
  parser = ArgumentParser("scene")
  parser.add_argument("--model-name", type=str, default="scene.pkl")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--gpu", type=int, default=-1)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--max-det", type=int, default=10)
  parser.add_argument("--sample-count", type=int, default=100)
  parser.add_argument("--learning-rate", type=float, default=5e-4)
  parser.add_argument("--cuda", action="store_true")
  args = parser.parse_args()


  accuracies = ["A " + str(i+1) for i in range(args.n_epochs)]
  times = ["T " + str(i+1) for i in range(args.n_epochs)]
  losses = ["L " + str(i+1) for i in range(args.n_epochs)]
  field_names = ['random seed', 'sample_count', 'grad_type'] + accuracies + times + losses

  for seed in [3177, 5848, 9175, 8725, 1234, 1357, 2468, 548, 6787, 8371]:
    torch.manual_seed(seed)
    random.seed(seed)

    data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data/scene"))
    model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/scene/ised"))
    if not os.path.exists(model_dir): os.makedirs(model_dir)
            
    (train_loader, test_loader) = scene_loader(data_root, args.batch_size)
    trainer = Trainer(train_loader, test_loader, args.learning_rate, args.sample_count, args.max_det, model_dir, seed)

    dict = trainer.train(args.n_epochs)
    dict["random seed"] = seed
    dict["sample_count"] = args.sample_count
    dict["grad_type"] = 'ised'
    with open('scene/bbox.csv', 'a', newline='') as csvfile:
          writer = csv.DictWriter(csvfile, fieldnames=field_names)
          writer.writerow(dict)
          csvfile.close()