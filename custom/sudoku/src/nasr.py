import os
import argparse
import torch.nn as nn
import torch
import math
from sudoku_solver.board import Board
import json
from time import time
from datasets import SudokuDataset_RL
import numpy as np
from torch.distributions.bernoulli import Bernoulli
import torch.nn.functional as F
from models.transformer_sudoku import get_model
try:
    from pyswip import Prolog
except Exception:
    print('-->> Prolog not installed')
SOLVER_TIME_OUT = 0.5

def init_parser():
    parser = argparse.ArgumentParser(description='Quick training script')

    # General args
    parser.add_argument('--gpu-id', default=0, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='log frequency (by iteration)')
    parser.add_argument('--solver', type=str, default='prolog',
                        help='symbolic solver to use. available options prolog and backtrack')
                        
    # Model args
    parser.add_argument('--block-len', default=81, type=int,
                        help='board size')
    parser.add_argument('--data', type=str, default='satnet',
                        help='dataset name between big_kaggle, minimal_17, multiple_sol and satnet')
    parser.add_argument('--noise-setting', default='xxx/yyy.json', type=str,
                        help='Json file of noise setting (dict)')
    parser.add_argument('--train-only-mask', default = False, type = bool,
                        help='If true, use RL to train only the mask predictor')

    # Optimization hyperparams
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--warmup', default=10, type=int, 
                        help='number of warmup epochs')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        help='mini-batch size (default: 128)', dest='batch_size')
    parser.add_argument('--lr', default=0.00001, type=float, 
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=3e-1, type=float, 
                        help='weight decay (default: 3e-2)')
    parser.add_argument('--clip-grad-norm', default=1., type=float, 
                        help='gradient norm clipping (default: 0 (disabled))')
    parser.add_argument('--disable-cos', action='store_true',
                        help='disable cosine lr schedule')
    return parser

def compute_reward(solution_board,final_solution,ground_truth_board):
    solution_board = list(map(int,solution_board.tolist()))
    final_solution = list(map(int, final_solution.tolist() ))
    ground_truth_board = list(map(int,ground_truth_board.tolist()))
    if final_solution == ground_truth_board:
        reward = 10
    else:
        reward = 0
    partial_reward = 0.0
    for i,j in zip(final_solution,ground_truth_board):
        if i == j:
            partial_reward+=1
    reward += partial_reward/81
    return reward

def final_output(model,ground_truth_sol,solution_boards,masking_boards,args):
    ground_truth_boards = torch.argmax(ground_truth_sol,dim=2)
    solution_boards_new = torch.argmax(solution_boards,dim=2)+1
    
    config = 'sigmoid_bernoulli' # best option
    # between sigmoid_bernoulli gumble_round sigmoid_round 
    if config == 'sigmoid_bernoulli':
        masking_prob = masking_boards.sigmoid()
        b = Bernoulli(masking_prob)
        sampled_mask_boards = b.sample()
        model.saved_log_probs = b.log_prob(sampled_mask_boards)
        sampled_mask_boards = np.array(sampled_mask_boards.cpu()).reshape(masking_prob.shape)
        cleaned_boards = np.multiply(solution_boards_new.cpu(),sampled_mask_boards)
    elif config == 'sigmoid_round':
        masking_prob = masking_boards.sigmoid()
        b= Bernoulli(masking_prob)
        sampled_mask_boards = torch.round(masking_prob)
        model.saved_log_probs = b.log_prob(sampled_mask_boards)
        cleaned_boards = np.multiply(solution_boards_new.cpu(),sampled_mask_boards.cpu())
    else: 
        assert(config == 'gumble_round')
        masking_prob = F.gumbel_softmax(masking_boards)
        b = Bernoulli(masking_prob)
        sampled_mask_boards = torch.round(masking_prob)
        model.saved_log_probs = b.log_prob(sampled_mask_boards)
        cleaned_boards = np.multiply(solution_boards_new.cpu(),sampled_mask_boards.cpu())
    
    final_boards = []
    if args.solver == 'prolog':
        prolog_instance = Prolog()
        prolog_instance.consult("src/sudoku_solver/sudoku_prolog.pl")
    for i in range(len(cleaned_boards)):
        board_to_solver = Board(cleaned_boards[i].reshape((9,9)).int())
        try:
            if args.solver == 'prolog':
                solver_success = board_to_solver.solve(solver ='prolog',prolog_instance = prolog_instance)
            else:
                solver_success = board_to_solver.solve(solver ='backtrack')
        except StopIteration:
            solver_success = False
        final_solution = board_to_solver.board.reshape(81,)
        if not solver_success:
            final_solution = solution_boards_new[i].cpu()
        reward = compute_reward(solution_boards_new[i].cpu(),final_solution,ground_truth_boards[i])
        model.rewards.append(reward)
        final_boards.append(final_solution)
    return final_boards

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    elif not args.disable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def validate(val_loader, model, args, epoch=None, time_begin=None):
    model.eval()
    loss_value = 0
    reward_value = 0
    n = 0
    eps = np.finfo(np.float32).eps.item()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(args.gpu_id)
            target = target.to(args.gpu_id)

            solution_boards, masking_boards = model(images)
            final_output(model,target,solution_boards,masking_boards,args) # this populates model.rewards 
            rewards = np.array(model.rewards)
            rewards_mean = rewards.mean()
            reward_value += float(rewards_mean * images.size(0))
            rewards = (rewards - rewards.mean())/(rewards.std() + eps)
            policy_loss = []
            for reward, log_prob in zip(rewards, model.saved_log_probs):
                policy_loss.append(-log_prob*reward)
            policy_loss = (torch.cat(policy_loss)).sum()

            n += images.size(0)
            loss_value += float(policy_loss.item() * images.size(0))
            model.rewards = []
            model.saved_log_probs = []
            torch.cuda.empty_cache()

            if args.print_freq >= 0 and i % args.print_freq == 0:
                avg_loss = (loss_value / n)
                print(f'[rl][Epoch {epoch}][Val][{i}] \t AvgLoss: {avg_loss:.4f}  \t AvgRewards: {rewards_mean:.4f}')
    
    avg_reward = (reward_value/n)      
    avg_loss = (loss_value / n)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(f'----[rl][Epoch {epoch}] \t \t AvgLoss {avg_loss:.4f} \t \t AvgReward {avg_reward:.4f} \t \t Time: {total_mins:.2f} ')

    return avg_loss, rewards_mean
    
def train(train_loader, model, optimizer, epoch, args):
    model.train()
    loss_value = 0
    n = 0
    
    # to train only the mask predictor
    if args.train_only_mask == True:
        # to not train nn_solver
        model.nn_solver.requires_grad_(requires_grad = False)
        for param in model.nn_solver.parameters():
            param.requires_grad = False
        # to not train perception
        model.perception.requires_grad_(requires_grad = False)
        for param in model.perception.parameters():
            param.requires_grad = False

    eps = np.finfo(np.float32).eps.item()
    for i, (images, target) in enumerate(train_loader):        
        images = images.to(args.gpu_id)
        target = target.to(args.gpu_id)
        solution_boards, masking_boards = model(images)
        final_output(model,target,solution_boards,masking_boards,args) # this populates model.rewards 
        rewards = np.array(model.rewards)
        rewards_mean = rewards.mean()
        rewards = (rewards - rewards.mean())/(rewards.std() + eps)
        policy_loss = []
        for reward, log_prob in zip(rewards, model.saved_log_probs):
            policy_loss.append(-log_prob*reward)
        optimizer.zero_grad()
        
        criterion = nn.BCEWithLogitsLoss()
        loss_nn_solver = criterion(solution_boards, target[:,:,1:])
        #policy_loss = (torch.cat(policy_loss)).sum() + loss_nn_solver
        policy_loss = (torch.cat(policy_loss)).sum()

        n += images.size(0)
        loss_value += float(policy_loss.item() * images.size(0))
        policy_loss.backward()
       
        #if args.clip_grad_norm > 0:
        #    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm, norm_type=2)

        optimizer.step()
             
        if args.print_freq >= 0 and i % args.print_freq == 0:
            avg_loss = (loss_value / n)
            print(f'[rl][Epoch {epoch}][Train][{i}/{len(train_loader)}] \t AvgLoss: {avg_loss:.4f} \t AvgRewards: {rewards_mean:.4f}')
        model.rewards = []
        model.saved_log_probs = []
        torch.cuda.empty_cache()

    avg_loss = (loss_value / n)
    return avg_loss, rewards_mean


def main():
    parser = init_parser()
    args = parser.parse_args()
    seed = 1234

    torch.manual_seed(seed)

    train_dataset = SudokuDataset_RL(args.data,'-train')
    val_dataset = SudokuDataset_RL(args.data,'-valid')
        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers)
        
    # Model 
    model = get_model(block_len=args.block_len)
    model.load_pretrained_models(args.data)
    model.to(args.gpu_id)

    # load pre_trained models
    if args.train_only_mask == True:
        # only training the mask network
        optimizer = torch.optim.AdamW(model.mask_nn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Main loop 
    print("Beginning training")
    ckpt_path = os.path.join('checkpoint', 'nasr')
    os.makedirs(ckpt_path, exist_ok=True)
    best_loss = None
    best_reward = None
    time_begin = time()
    with open(f"{ckpt_path}/log_{seed}.txt", 'w'): pass
    for epoch in range(args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, args)

        train_loss, train_rewards = train(train_loader, model, optimizer, epoch, args)
        loss, valid_rewards = validate(val_loader, model, args, epoch=epoch, time_begin=time_begin)
            
        if best_reward is None or valid_rewards > best_reward :
            best_reward = valid_rewards
            torch.save(model.state_dict(), f'{ckpt_path}/checkpoint_best_R_{seed}.pth')
            
        if best_loss is None or loss < best_loss :
            best_loss = loss
            torch.save(model.state_dict(), f'{ckpt_path}/checkpoint_best_L_{seed}.pth')

        stats = {'epoch': epoch, 'lr': lr, 'train_loss': train_loss, 
                 'val_loss': loss, 'best_loss': best_loss , 
                 'train_rewards': train_rewards, 'valid_rewards': valid_rewards}
        with open(f"{ckpt_path}/log_{seed}.txt", "a") as f:
            f.write(json.dumps(stats) + "\n")
        torch.save(model.state_dict(), f'{ckpt_path}/checkpoint_{epoch}_{seed}.pth')

    total_mins = (time() - time_begin) / 60
    print(f'[rl] finished in {total_mins:.2f} minutes, '
            f'best loss: {best_loss:.6f}, '
            f'final loss: {loss:.6f}')

if __name__ == '__main__':

    main()