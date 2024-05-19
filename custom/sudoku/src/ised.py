import torch
import torch.nn.functional as F
import torch.nn as nn

import math
import os
import json
import numpy as np
from time import time

from argparse import ArgumentParser

from datasets import SudokuDataset_RL
from models.transformer_sudoku import get_model
from sudoku_solver.board import Board

try:
    from pyswip import Prolog
except Exception:
    print('-->> Prolog not installed')

def vectorize(results, sample_probs):
    batch_size, _ = sample_probs.shape
    size = results.shape[2]
    n_digits = int(math.sqrt(size))
    result_tensor = torch.zeros((batch_size, size, n_digits), device='cuda')
    for i, result in enumerate(results):
        for j, r in enumerate(result):
            result_prob = sample_probs[i][j]
            for k in range(size):
                sampled_digit = int(r[k].item())
                if sampled_digit != 0:
                    idx = sampled_digit - 1
                    result_tensor[i][k][idx] += result_prob
    return torch.nn.functional.normalize(result_tensor,dim=2)

def compute_loss(final_solution,ground_truth_board,sample_probs):
    # vectorize final_solution
    vectorized = vectorize(final_solution,sample_probs).to("cuda")
    # vectorize ground_truth_board
    ground_truth_idxs = ground_truth_board-1
    ground_truth_one_hot = torch.nn.functional.one_hot(ground_truth_idxs,num_classes=9).float()
    return torch.nn.functional.binary_cross_entropy(vectorized,ground_truth_one_hot)

def bbox(ground_truth_sol,solution_boards,masking_boards,args):
  ground_truth_boards = torch.argmax(ground_truth_sol,dim=2)

  sample_count = args.sample_count
  distrs = torch.distributions.Categorical(probs=solution_boards)
  sampled_boards = distrs.sample((sample_count,)) + 1

  masking_prob = masking_boards.sigmoid()
  b = torch.distributions.bernoulli.Bernoulli(masking_prob.repeat(sample_count,1,1))
  sampled_mask_boards = b.sample()
  sampled_mask_boards = np.array(sampled_mask_boards.cpu()).reshape(masking_prob.repeat(sample_count,1,1).shape)
  sampled_mask_boards_expanded = torch.from_numpy(sampled_mask_boards)
  cleaned_sampled_boards = np.multiply(sampled_boards.cpu(),sampled_mask_boards_expanded).permute(1, 0, 2) # 256 2 81
  
  final_boards = []
  for i in range(len(cleaned_sampled_boards)):
    # each i corresponds to one item in the batch
    final_boards_i = []
    for j in range(len(cleaned_sampled_boards[i])):
      # each j corresponds to one sample in cleaned_sampled_boards[i]
      board_to_solver = Board(cleaned_sampled_boards[i][j].reshape((9, 9)).int())
      try:
        if args.solver == 'prolog':
          prolog_instance = Prolog()
          dir_path = os.path.dirname(os.path.realpath(__file__))
          prolog_instance.consult(dir_path + "/sudoku_solver/sudoku_prolog.pl")
          solver_success = board_to_solver.solve(solver ='prolog',prolog_instance = prolog_instance)
        else:
          solver_success = board_to_solver.solve(solver ='backtrack')
      except StopIteration:
        solver_success = False
      final_solution = board_to_solver.board.reshape(81,)
      if not solver_success:
        final_solution_tensor = cleaned_sampled_boards[i][j].cpu()
      else:
        final_solution_tensor = torch.from_numpy(final_solution)
      final_boards_i.append(final_solution_tensor)
    final_boards.append(torch.stack(final_boards_i))

  cleaned_sampled_boards_swapped = cleaned_sampled_boards.permute(0, 2, 1)
  masking_prob = masking_prob.unsqueeze(-1)
  boards_prob = torch.cat((1-masking_prob, (masking_prob * solution_boards)), dim=-1).cpu()
  gathered_probs = boards_prob.gather(2, cleaned_sampled_boards_swapped.long())
  
  sample_probs = torch.prod(gathered_probs, dim=1)
  final_boards = torch.stack(final_boards)
  loss = compute_loss(final_boards,ground_truth_boards,sample_probs)
  return loss

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    elif not args.disable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
    
class Trainer():
  def __init__(self, train_loader, test_loader, model, path, seed, args):
    self.network = model
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.path = path
    self.args = args
    self.best_loss = None
    self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    self.seed = seed

  def loss(self, output, ground_truth):
    (_, dim) = output.shape
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
    return F.binary_cross_entropy(output, gt) # BCEWithLogitsLoss
  
  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    train_loss = 0

    for i, (images, target) in enumerate(self.train_loader):
      images = images.to(self.args.gpu_id)
      target = target.to(self.args.gpu_id)
      solution_boards, masking_boards = self.network(images)
      loss = bbox(target, solution_boards, masking_boards, self.args)
      self.optimizer.zero_grad()
      loss.backward()

      if args.clip_grad_norm > 0:
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=args.clip_grad_norm, norm_type=2)
    
      self.optimizer.step()

      num_items += images.size(0)
      train_loss += loss.item()

      torch.cuda.empty_cache() 
      
      if args.print_freq >= 0 and i % args.print_freq == 0:
        avg_loss = train_loss/num_items
        print(f'[rl][Epoch {epoch}][Train][{i}/{len(train_loader)}] \t AvgLoss: {avg_loss:.4f}')
    
    return train_loss/num_items

  def test_epoch(self, epoch, time_begin):
    self.network.eval()
    num_items = 0
    test_loss = 0

    with torch.no_grad():
      for i, (images, target) in enumerate(self.test_loader):
        images = images.to(self.args.gpu_id)
        target = target.to(self.args.gpu_id)
        solution_boards, masking_boards = self.network(images)
        loss = bbox(target, solution_boards, masking_boards, self.args)

        num_items += images.size(0)
        test_loss += loss.item()
        torch.cuda.empty_cache()

        if args.print_freq >= 0 and i % args.print_freq == 0:
          avg_loss = (test_loss / num_items)
          print(f'[rl][Epoch {epoch}][Val][{i}] \t AvgLoss: {avg_loss:.4f}')

      if self.best_loss is None or test_loss < self.best_loss:
          self.best_loss = test_loss
          # torch.save(self.network.state_dict(), f'{self.path}/checkpoint_best_L_{self.seed}.pth')
    
    avg_loss = (test_loss / num_items)
    total_mins = (time() - time_begin) / 60
    print(f'----[rl][Epoch {epoch}] \t \t AvgLoss {avg_loss:.4f} \t \t Time: {total_mins:.2f} ')
    return avg_loss

  def train(self, n_epochs):
    time_begin = time()
    with open(f"{self.path}/log_{self.seed}.txt", 'w'): pass
    for epoch in range(1, n_epochs+1):
      lr = adjust_learning_rate(self.optimizer, epoch, self.args)
      train_loss = self.train_epoch(epoch)
      test_loss = self.test_epoch(epoch, time_begin)
      stats = {'epoch': epoch, 'lr': lr, 'train_loss': train_loss, 'val_loss': test_loss, 'best_loss': self.best_loss}
      with open(f"{self.path}/log_{self.seed}.txt", "a") as f: 
        f.write(json.dumps(stats) + "\n")
      
      torch.save(self.network.state_dict(), f'{self.path}/checkpoint_{self.seed}_{epoch}.pth')

if __name__ == "__main__":
  parser = ArgumentParser('sudoku')
  parser.add_argument('--gpu-id', default=0, type=int)
  parser.add_argument('-j', '--workers', default=4, type=int)
  parser.add_argument('--print-freq', default=10, type=int)
  parser.add_argument('--solver', type=str, default='prolog')

  parser.add_argument('--block-len', default=81, type=int)
  parser.add_argument('--data', type=str, default='satnet')
  parser.add_argument('--noise-setting', default='xxx/yyy.json', type=str)
  parser.add_argument('--train-only-mask', default = False, type = bool)

  parser.add_argument('--epochs', default=5, type=int)
  parser.add_argument('--seed', default=1234, type=int)
  parser.add_argument('--warmup', default=10, type=int)
  parser.add_argument('-b', '--batch-size', default=256, type=int)
  parser.add_argument('--lr', default=0.00001, type=float)
  parser.add_argument('--weight-decay', default=3e-1, type=float)
  parser.add_argument('--clip-grad-norm', default=1., type=float)
  parser.add_argument('--disable-cos', action='store_true')
  parser.add_argument('--sample-count', default=3, type=int)

  args = parser.parse_args()

  torch.manual_seed(args.seed)

  train_dataset = SudokuDataset_RL(args.data,'-train')
  test_dataset = SudokuDataset_RL(args.data,'-valid')

  # Model
  model = get_model(block_len=args.block_len)
  model.load_pretrained_models(args.data)
  model.to(args.gpu_id)

  model_dir = os.path.join('checkpoint', 'ised')
  os.makedirs(model_dir, exist_ok=True)

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

  # load pre_trained models
  trainer = Trainer(train_loader, test_loader, model, model_dir, args.seed, args)
  trainer.train(args.epochs)
