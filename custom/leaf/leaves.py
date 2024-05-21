import os
import random
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

from argparse import ArgumentParser

import blackbox
from leaves_config import leaves_loader, LeafNet, l11_labels, l11_margin, l11_shape, l11_texture, classify_11

class LeavesNet(nn.Module):
  def __init__(self, sample_count, caching):
    super(LeavesNet, self).__init__()

    # features for classification
    self.f1 = l11_margin
    self.f2 = l11_shape
    self.f3 = l11_texture
    self.labels = l11_labels

    self.net1 = LeafNet(len(self.f1))
    self.net2 = LeafNet(len(self.f2))
    self.net3 = LeafNet(len(self.f3))

    # Blackbox encoding identification chart
    self.bbox = blackbox.BlackBoxFunction(
                classify_11,
                (blackbox.DiscreteInputMapping(self.f1),
                 blackbox.DiscreteInputMapping(self.f2),
                 blackbox.DiscreteInputMapping(self.f3)),
                 blackbox.DiscreteOutputMapping(self.labels),
                caching=caching,
                sample_count=sample_count)

  def forward(self, x):
    has_f1 = self.net1(x)
    has_f2 = self.net2(x)
    has_f3 = self.net3(x)
    return self.bbox(has_f1, has_f2, has_f3)

class Trainer():
  def __init__(self, train_loader, test_loader, learning_rate, sample_count, model_dir, caching, gpu, seed, save_model=True):
    if gpu >= 0:
      device = torch.device("cuda:%d" % gpu)
    else:
      device = torch.device("cpu")
    self.device = device
    self.network = LeavesNet(sample_count, caching) #.to(self.device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.save_model = save_model
    self.model_dir = model_dir
    self.seed = seed

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
    train_loss = 0
    for (i, (input, target)) in enumerate(self.train_loader):
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
      train_loss += loss.item()
      avg_loss = train_loss / (i + 1)
    print(f"[Train Epoch {epoch}] Loss: {avg_loss:.4f}, Overall Accuracy: {int(total_correct)}/{int(num_items)} ({correct_perc:.2f})%")
    return train_loss

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = 0
    num_correct = 0
    test_loss = 0
    with torch.no_grad():
      for i, (input, target) in enumerate(test_loader):
        input = input.to(self.device)
        target = target.to(self.device)
        output = self.network(input)
        num_items += output.shape[0]
        test_loss += self.loss_fn(output, target).item()
        num_correct += (output.argmax(dim=1)==target).float().sum()
        perc = 100.*num_correct/num_items
        avg_loss = test_loss / (i + 1)
      print(f"[Test Epoch {epoch}] Avg loss: {avg_loss:.4f}, {int(num_correct)}/{int(num_items)} ({perc:.2f})%")
    
    # Save the model
    if self.save_model and test_loss < self.min_test_loss:
      self.min_test_loss = test_loss
      torch.save(self.network.state_dict(), self.model_dir + f"/ised_{self.seed}_best.pth")
    
    return float(num_correct/num_items)

  def train(self, n_epochs):
    for epoch in range(1, n_epochs+1):
      t0 = time.time()
      train_loss = self.train_epoch(epoch)
      t1 = time.time()
      acc = self.test_epoch(epoch)

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("leaves")
  parser.add_argument("--model-name", type=str, default="leaves.pkl")
  parser.add_argument("--n-epochs", type=int, default=30)
  parser.add_argument('--seed', default=1234, type=int)
  parser.add_argument("--sample-count", type=int, default=100)
  parser.add_argument("--gpu", type=int, default=-1)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--caching", type=bool, default=False)
  parser.add_argument("--cuda", action="store_true")
  args = parser.parse_args()

  train_num = 30
  test_nums = 10
  data_dir = 'leaf_11'

  # Setup parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Load data
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/leaves"))
  if not os.path.exists(model_dir): os.makedirs(model_dir)
        
  (train_loader, test_loader) = leaves_loader(data_root, data_dir, train_num, args.batch_size, test_nums)
  trainer = Trainer(train_loader, test_loader, args.learning_rate, args.sample_count, model_dir, args.caching, args.gpu, args.seed)

  # Run
  trainer.train(args.n_epochs)