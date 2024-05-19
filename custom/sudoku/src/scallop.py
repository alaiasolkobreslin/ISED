import torch
import torch.nn.functional as F
from torch import nn

import math
import os
import json
from time import time
from typing import Optional, Callable
import scallopy

from argparse import ArgumentParser

from datasets import SudokuDataset_RL
from models.transformer_sudoku import get_model

class SudokuNet(nn.Module):
    def __init__(self, provenance):
        super(SudokuNet, self).__init__()
        self.network = get_model(block_len=81)
        self.network.load_pretrained_models('big_kaggle')
        
        self.base_ctx = scallopy.Context(provenance = provenance, k=1)
        self.base_ctx.import_file("src/sudoku_solver/sudoku_scallop.scl")
        self.base_ctx.add_relation("digit_1", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_2", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_3", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_4", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_5", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_6", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_7", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_8", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_9", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_10", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_11", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_12", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_13", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_14", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_15", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_16", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_17", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_18", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_19", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_20", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_21", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_22", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_23", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_24", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_25", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_26", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_27", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_28", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_29", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_30", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_31", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_32", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_33", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_34", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_35", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_36", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_37", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_38", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_39", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_40", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_41", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_42", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_43", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_44", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_45", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_46", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_47", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_48", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_49", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_50", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_51", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_52", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_53", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_54", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_55", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_56", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_57", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_58", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_59", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_60", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_61", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_62", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_63", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_64", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_65", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_66", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_67", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_68", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_69", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_70", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_71", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_72", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_73", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_74", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_75", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_76", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_77", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_78", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_79", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_80", int, input_mapping=list(range(10)))
        self.base_ctx.add_relation("digit_81", int, input_mapping=list(range(10)))
        self.base_ctx.relation_is_non_probabilistic("distinct_nine")
        self.base_ctx.relation_is_non_probabilistic("op")
        self.base_ctx.relation_is_non_probabilistic("board")
        self.predict = self.base_ctx.forward_function("get_prediction", dispatch="parallel")
    
    def forward(self, x):
        solution_boards, masking_boards = self.network(x)
        masking_probs = masking_boards.sigmoid().unsqueeze(-1)
        merged = torch.cat((1-masking_probs, masking_probs*solution_boards), dim = -1)
        r = self.predict(
          digit_1 = merged[:,0], digit_2 = merged[:,1], digit_3 = merged[:,2], digit_4 = merged[:,3], digit_5 = merged[:,4], digit_6 = merged[:,5], digit_7 = merged[:,6], 
          digit_8 = merged[:,7], digit_9 = merged[:,8], digit_10 = merged[:,9], digit_11 = merged[:,10], digit_12 = merged[:,11], digit_13 = merged[:,12], 
          digit_14 = merged[:,13], digit_15 = merged[:,14], digit_16 = merged[:,15], digit_17 = merged[:,16], digit_18 = merged[:,17], digit_19 = merged[:,18], 
          digit_20 = merged[:,19], digit_21 = merged[:,20], digit_22 = merged[:,21], digit_23 = merged[:,22], digit_24 = merged[:,23], digit_25 = merged[:,24], 
          digit_26 = merged[:,25], digit_27 = merged[:,26], digit_28 = merged[:,27], digit_29 = merged[:,28], digit_30 = merged[:,29], digit_31 = merged[:,30], 
          digit_32 = merged[:,31], digit_33 = merged[:,32], digit_34 = merged[:,33], digit_35 = merged[:,34], digit_36 = merged[:,35], digit_37 = merged[:,36], 
          digit_38 = merged[:,37], digit_39 = merged[:,38], digit_40 = merged[:,39], digit_41 = merged[:,40], digit_42 = merged[:,41], digit_43 = merged[:,42], 
          digit_44 = merged[:,43], digit_45 = merged[:,44], digit_46 = merged[:,45], digit_47 = merged[:,46], digit_48 = merged[:,47], digit_49 = merged[:,48], 
          digit_50 = merged[:,49], digit_51 = merged[:,50], digit_52 = merged[:,51], digit_53 = merged[:,52], digit_54 = merged[:,53], digit_55 = merged[:,54], 
          digit_56 = merged[:,55], digit_57 = merged[:,56], digit_58 = merged[:,57], digit_59 = merged[:,58], digit_60 = merged[:,59], digit_61 = merged[:,60], 
          digit_62 = merged[:,61], digit_63 = merged[:,62], digit_64 = merged[:,63], digit_65 = merged[:,64], digit_66 = merged[:,65], digit_67 = merged[:,66], 
          digit_68 = merged[:,67], digit_69 = merged[:,68], digit_70 = merged[:,69], digit_71 = merged[:,70], digit_72 = merged[:,71], digit_73 = merged[:,72], 
          digit_74 = merged[:,73], digit_75 = merged[:,74], digit_76 = merged[:,75], digit_77 = merged[:,76], digit_78 = merged[:,77], digit_79 = merged[:,78], 
          digit_80 = merged[:,79], digit_81 = merged[:,80])
        return r

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
  def __init__(self, train_loader, test_loader, path, args):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.network = SudokuNet(args.provenance).to(self.device)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.path = path
    self.args = args
    self.best_loss = None
    self.best_acc = None
    self.loss_fn = F.binary_cross_entropy
    self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=args.lr, weight_decay=args.weight_decay)

  def loss(self, y_pred_values, y_pred_probs, ground_truth):
    gt = torch.stack([torch.tensor([(torch.tensor(i).to(self.device) == t).sum() for i in y_pred_values]) for t in ground_truth]).to(self.device)
    gt = gt/(gt.sum(dim=-1).unsqueeze(-1))
    return F.binary_cross_entropy(y_pred_probs, gt)
  
  def acc(self, y_pred_values, y_pred_probs, ground_truth):
    preds = []
    for i in y_pred_probs.argmax(dim=-1):
       preds.append(torch.tensor(y_pred_values[i]))
    y_pred = torch.stack(preds).to(self.device)
    y = torch.where(ground_truth == y_pred, 1, 0)
    return y.sum()
  
  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    train_loss = 0
    total_correct = 0

    for i, (images, target) in enumerate(self.train_loader):
      images = images.to(self.device)
      target = target.to(self.device).argmax(dim=-1)
      y_pred_vals, y_pred_probs = self.network(images)
      loss = self.loss(y_pred_vals, y_pred_probs, target)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()  

      #if args.clip_grad_norm > 0:
      #  nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=args.clip_grad_norm, norm_type=2)

      total_correct += self.acc(y_pred_vals, y_pred_probs, target)
      num_items += images.size(0)
      train_loss += loss.item()

      torch.cuda.empty_cache()
      
      if args.print_freq >= 0 and i % args.print_freq == 0:
        avg_loss = train_loss/num_items
        avg_acc = total_correct/num_items
        print(f'[rl][Epoch {epoch}][Train][{i}/{len(train_loader)}] \t AvgLoss: {avg_loss:.4f} \t AvgAcc: {avg_acc:.4f}')
    
    return train_loss

  def test_epoch(self, epoch, time_begin):
    self.network.eval()
    num_items = 0
    test_loss = 0
    total_correct = 0

    with torch.no_grad():
      for i, (images, target) in enumerate(self.test_loader):
        images = images.to(self.args.gpu_id)
        target = target.to(self.args.gpu_id)
        y_pred_vals, y_pred_probs = self.network(images)
        loss = self.loss(y_pred_vals, y_pred_probs, target)

        total_correct += self.acc(y_pred_vals, y_pred_probs, target)
        num_items += images.size(0)
        test_loss += loss.item()
        
        ttorch.cuda.empty_cache()

        if args.print_freq >= 0 and i % args.print_freq == 0:
          avg_loss = (test_loss / num_items)
          avg_acc = total_correct/num_items
          print(f'[rl][Epoch {epoch}][Val][{i}] \t AvgLoss: {avg_loss:.4f} \t AvgAcc: {avg_acc:.4f}')

    if self.best_loss is None or test_loss < self.best_loss:
        self.best_loss = test_loss
        torch.save(self.network.state_dict(), f'{self.path}/checkpoint_best_L.pth')

    if self.best_acc is None or total_correct > self.best_acc:
        self.best_loss = total_correct
        torch.save(self.network.state_dict(), f'{self.path}/checkpoint_best_R.pth')

    avg_loss = (test_loss / num_items)
    total_mins = (time() - time_begin) / 60
    print(f'----[rl][Epoch {epoch}] \t \t AvgLoss {avg_loss:.4f} \t \t Time: {total_mins:.2f} ')
    
    return test_loss, float(total_correct / num_items)

  def train(self, n_epochs):
    time_begin = time()
    for epoch in range(1, n_epochs+1):
      lr = adjust_learning_rate(self.optimizer, epoch, self.args)
      train_loss = self.train_epoch(epoch)
      test_loss, test_acc = self.test_epoch(epoch, time_begin)
      stats = {'epoch': epoch, 'lr': lr, 'train_loss': train_loss, 'val_loss': test_loss, 'val_acc': test_acc, 'best_loss': self.best_loss}
      with open(f"{self.path}/log.txt", "a") as f: 
        f.write(json.dumps(stats) + "\n")
      
      torch.save(self.network.state_dict(), f'{self.path}/checkpoint_{epoch}.pth')

if __name__ == "__main__":
  parser = ArgumentParser('sudoku')
  parser.add_argument('-j', '--workers', default=0, type=int)
  parser.add_argument('--print-freq', default=10, type=int)
  parser.add_argument('--solver', type=str, default='prolog')

  parser.add_argument('--block-len', default=81, type=int)
  parser.add_argument('--data', type=str, default='big_kaggle')
  parser.add_argument('--noise-setting', default='xxx/yyy.json', type=str)
  parser.add_argument('--train-only-mask', default = False, type = bool)

  parser.add_argument('--epochs', default=1, type=int)
  parser.add_argument('--warmup', default=10, type=int)
  parser.add_argument('-b', '--batch-size', default=2, type=int)
  parser.add_argument('--lr', default=0.00001, type=float)
  parser.add_argument('--weight-decay', default=3e-1, type=float)
  parser.add_argument('--clip-grad-norm', default=1., type=float)
  parser.add_argument('--disable-cos', action='store_true')

  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--top-k", type=int, default=1)
  parser.add_argument("--jit", action="store_true")
  parser.add_argument("--dispatch", type=str, default="parallel")

  args = parser.parse_args()

  torch.manual_seed(3177)

  train_dataset = SudokuDataset_RL(args.data,'-train')
  test_dataset = SudokuDataset_RL(args.data,'-valid')

  model_dir = os.path.join('outputs', 'scallop')
  os.makedirs(model_dir, exist_ok=True)

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

  # load pre_trained models
  trainer = Trainer(train_loader, test_loader, model_dir, args)
  trainer.train(args.epochs)