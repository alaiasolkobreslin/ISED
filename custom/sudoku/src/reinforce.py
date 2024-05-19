import math
import os
import time
import torch

from argparse import ArgumentParser

from datasets import SudokuDataset_RL
from models.transformer_sudoku import get_model
from sudoku_solver.board import Board

try:
    from pyswip import Prolog
except Exception:
    print('-->> Prolog not installed')

def compute_reward(final_solution,ground_truth_board):
    final_solution = list(map(int, final_solution))
    ground_truth_board = list(map(int,ground_truth_board))
    partial_reward = 0.0
    for i,j in zip(final_solution,ground_truth_board):
        if i == j:
            partial_reward+=1
    return partial_reward/81

def loss_fn(clean_boards, solution_boards_new, ground_truth_boards):
  final_boards = []
  rewards = []
  prolog_instance = Prolog()
  prolog_instance.consult("src/sudoku_solver/sudoku_prolog.pl") 
  cleaned_boards = clean_boards.flatten(0,-2)
  ground_truth_boards = ground_truth_boards.flatten(0,-2)
  solution_boards_new = solution_boards_new.flatten(0,-2)
  for i in range(len(cleaned_boards)):
    board_to_solver = Board(cleaned_boards[i].reshape((9,9)).int().cpu())
    try:
      solver_success = board_to_solver.solve(solver ='prolog',prolog_instance = prolog_instance)
    except StopIteration:
      solver_success = False
    final_solution = board_to_solver.board.reshape(81,)
    if not solver_success:
      final_solution = solution_boards_new[i].cpu()
    reward = compute_reward(final_solution,ground_truth_boards[i])
    rewards.append(torch.tensor(reward))
    final_boards.append(final_solution)
  return torch.stack(rewards).reshape(clean_boards.shape[:-1]), final_boards

class Trainer():
  def __init__(self, model, loss_fn, train_loader, test_loader, model_dir, learning_rate, grad_type, dim, sample_count, batch_size, log_it, seed, args):
    self.model_dir = model_dir
    self.network = model
    self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = None
    self.best_reward = None
    self.grad_type = grad_type
    self.dim = dim
    self.sample_count = sample_count
    self.batch_size = batch_size
    self.loss_fn = loss_fn
    self.log_it = log_it
    self.args = args
    self.seed = seed
    self.device = torch.device("CUDA")

  def reinforce_grads(self, data, target):
    solution_boards, masking_boards = self.network(data)
    ground_truth_boards = torch.argmax(target,dim=2)
    solution_boards_new = torch.argmax(solution_boards,dim=2)+1
    masking_prob = masking_boards.sigmoid()

    d = torch.distributions.Bernoulli(logits=masking_prob)
    samples = d.sample((self.sample_count,))
    cleaned_boards = (solution_boards_new) * samples

    f_sample, _ = self.loss_fn(cleaned_boards, solution_boards_new.repeat(self.sample_count,1), ground_truth_boards.repeat(self.sample_count,1))
    f_mean = f_sample.mean(dim=0)

    log_p_sample = d.log_prob(samples).sum(dim=-1).cpu()
    reinforce = (f_sample.detach() * log_p_sample).mean(dim=0)
    reinforce_prob = (f_mean - reinforce).detach() + reinforce
    loss = -torch.log(reinforce_prob + 1e-8)
    loss = loss.mean(dim=0)
    return f_mean.mean(dim=0), loss
  
  def indecater_grads(self, data, target):
    solution_boards, masking_boards = self.network(data)
    ground_truth_boards = torch.argmax(target,dim=2)
    solution_boards_new = torch.argmax(solution_boards,dim=2)+1
    masking_prob = masking_boards.sigmoid()
    
    d = torch.distributions.Bernoulli(logits=masking_prob)
    outer_samples = d.sample((self.sample_count,))
    outer_clean_boards = outer_samples * solution_boards_new
    solution_boards_new = solution_boards_new.repeat(self.sample_count,1,1)
    ground_truth_boards = ground_truth_boards.repeat(self.sample_count,1,1)
    outer_loss, _ = self.loss_fn(outer_clean_boards, solution_boards_new, ground_truth_boards)
    
    variable_loss = outer_loss.mean(dim=0).unsqueeze(-1).unsqueeze(-1)
    indecater_expression = variable_loss.detach() * masking_prob.cpu().unsqueeze(-1) * solution_boards.cpu()
    indecater_expression = indecater_expression.sum(dim=-1)
    indecater_expression = indecater_expression.sum(dim=-1)

    loss = -torch.log(indecater_expression + 1e-8)
    loss = loss.mean(dim=0)
    return outer_loss.mean(dim=0).mean(dim=0),loss
  
  def grads(self, data, target):
    if self.grad_type == 'reinforce':
      return self.reinforce_grads(data, target)
    elif self.grad_type == 'icr':
      return self.indecater_grads(data, target)
    
  def adjust_learning_rate(self, epoch):
    lr = self.args.lr
    if hasattr(self.args, 'warmup') and epoch < self.args.warmup:
        lr = lr / (self.args.warmup - epoch)
    elif not self.args.disable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - self.args.warmup) / (self.args.epochs - self.args.warmup)))
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr
    return lr

  def train_epoch(self, epoch):
    counter = 1
    self.network.train()
    for (data, target) in self.train_loader:
      self.optimizer.zero_grad()
      data = data.to(self.args.gpu_id)
      target = target.to(self.args.gpu_id)
      acc, loss = self.grads(data, target)
      loss.backward()
       
      self.optimizer.step()
      torch.cuda.empty_cache() 

      if counter % self.log_it == 0:
        print(f"Epoch {epoch} iterations {counter}",
              f"Reward {acc}")
      counter += 1

  def test_epoch(self, epoch):
    reward = 0
    loss = 0
    n = len(self.test_loader)
    self.network.eval()
    with torch.no_grad():
      for (data, target) in self.test_loader:
        data = data.to(self.args.gpu_id)
        target = target.to(self.args.gpu_id)
        r, l = self.grads(data, target)
        reward += float(r * data.shape[0])
        loss += float(l * data.shape[0])
        torch.cuda.empty_cache()
      avg_loss = loss/n
      avg_reward = reward/n 
      print(f'----[rl][Epoch {epoch}] \t \t AvgLoss {avg_loss:.4f} \t \t AvgReward {avg_reward:.4f}')

    if self.best_loss is None or loss < self.best_loss:
      self.best_loss = loss
      torch.save(self.network.state_dict(), f'{self.model_dir}/checkpoint_best_L_{self.seed}.pth')  

    if self.best_reward is None or reward > self.best_reward:
      self.best_reward = reward
      torch.save(self.network.state_dict(), f'{self.model_dir}/checkpoint_best_R_{self.seed}.pth')

  def train(self, n_epochs):
    # self.test_epoch(0)
    for epoch in range(1, n_epochs + 1):
      self.adjust_learning_rate(epoch)
      time1 = time.time()
      self.train_epoch(epoch)
      time2 = time.time()
      print(time2 - time1)
      time1 = time.time()
      self.test_epoch(epoch)
      time2 = time.time()
      print(time2 - time1)
      torch.save(model.state_dict(), f'{self.model_dir}/checkpoint_{self.seed}_{epoch}.pth')

if __name__ == "__main__":
  parser = ArgumentParser('sudoku_reinforce')
  parser.add_argument('--gpu-id', default=0, type=int)
  parser.add_argument('-j', '--workers', default=4, type=int)
  parser.add_argument('--solver', type=str, default='prolog')
  parser.add_argument('-j', '--workers', default=0, type=int) 

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
  parser.add_argument('--grad-type', type=str, default='reinforce')
  args = parser.parse_args()

  torch.manual_seed(args.seed)
  train_dataset = SudokuDataset_RL(args.data,'-train')
  test_dataset = SudokuDataset_RL(args.data,'-valid')

  # Model
  model = get_model(block_len=args.block_len)
  model.load_pretrained_models(args.data)
  model.to(args.gpu_id)

  model_dir = os.path.join('checkpoint', f'{args.grad_type}')
  os.makedirs(model_dir, exist_ok=True)

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

  # load pre_trained models
  trainer = Trainer(model, loss_fn, train_loader, test_loader, model_dir, args.lr, args.grad_type, args.block_len, args.sample_count, args.batch_size, args.print_freq, args.seed, args)
  trainer.train(args.epochs)