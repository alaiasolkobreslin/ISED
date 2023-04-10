import os
import json
import random
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from tqdm import tqdm

import unstructured_dataset
import task_dataset
import task_program
import sample
from constants import *


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, train):
        self.dataset = task_dataset.TaskDataset(config)
        self.train = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, _):
        return self.dataset.generate_datapoint(train=self.train)

    @staticmethod
    def collate_fn(batch):
        dicts = [item[0] for item in batch]
        imgs = torch.stack([torch.stack([item[0][k]
                           for item in batch]) for k in dicts[0].keys()])
        results = [item[1] for item in batch]
        return (imgs, results)


def train_test_loader(configuration, batch_size_train, batch_size_test):
    train_loader = torch.utils.data.DataLoader(
        Dataset(configuration, train=True),
        collate_fn=Dataset.collate_fn,
        batch_size=batch_size_train,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        Dataset(configuration, train=False),
        collate_fn=Dataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=True
    )

    return train_loader, test_loader


class TaskNet(nn.Module):
    def __init__(self, unstructured_datasets, fn):
        super(TaskNet, self).__init__()

        self.nets_dict = {}
        self.nets = []
        for ud in unstructured_datasets:
            if type(ud) is unstructured_dataset.MNISTDataset:
                if MNIST not in self.nets_dict:
                    self.nets_dict[MNIST] = ud.net()
                self.nets.append(self.nets_dict[MNIST])
            # TODO: finish

        n_inputs = len(self.nets)

        self.sampling = sample.Sample(n_inputs, args.n_samples, fn)

    def parameters(self):
        return [net.parameters() for net in self.nets_dict.values()]

    def task_test(self, args):
        return self.sampling.sample_test(args)

    def forward(self, x, y):
        n_inputs = len(x)
        distrs = [self.nets[i](x[i]) for i in range(n_inputs)]
        argss = list(zip(*(tuple(distrs)), y))
        out_pred = map(self.sampling.sample_train, argss)
        out_pred = list(zip(*out_pred))
        I_p, I_m = out_pred[0], out_pred[1]
        I_p = torch.stack(I_p).view(-1)
        I_m = torch.stack(I_m).view(-1)

        I = torch.cat((I_p, I_m))
        I_truth = torch.cat((torch.ones(size=I_p.shape, requires_grad=True), torch.zeros(
            size=I_m.shape, requires_grad=True)))

        l = F.mse_loss(I, I_truth)

        return l

    def evaluate(self, x):
        """
        Invoked during testing
        """
        n_inputs = len(x)
        distrs = [self.nets[i](x[i]) for i in range(n_inputs)]
        return self.task_test(distrs)


class Trainer():
    def __init__(self, train_loader, test_loader, learning_rate, unstructured_datasets, fn):
        self.network = TaskNet(unstructured_datasets, fn)
        self.optimizers = [optim.Adam(
            net.parameters(), lr=learning_rate) for net in self.network.nets_dict.values()]
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train_epoch(self, epoch):
        self.network.train()
        iter = tqdm(self.train_loader, total=len(self.train_loader))
        total_loss = 0.0
        for (batch_id, (data, target)) in enumerate(iter):
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            loss = self.network.forward(data, target)
            loss.backward()
            for parameters in self.network.parameters():
                for param in parameters:
                    param.grad.data.clamp_(-1, 1)
            for optimizer in self.optimizers:
                optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / (batch_id + 1)
            iter.set_description(
                f"[Train Epoch {epoch}] Avg Loss: {avg_loss:.4f}, Batch Loss: {loss.item():.4f}")

    def test_epoch(self, epoch):
        self.network.eval()
        num_items = 0
        correct = 0
        with torch.no_grad():
            iter = tqdm(self.test_loader, total=len(self.test_loader))
            for (data, target) in iter:
                batch_size = len(target)
                output = self.network.evaluate(data)
                for i in range(batch_size):
                    if output[i].item() == target[i]:
                        correct += 1
                num_items += batch_size
                perc = 100. * correct / num_items
                iter.set_description(
                    f"[Test Epoch {epoch}] Accuracy: {correct}/{num_items} ({perc:.2f}%)")

    def train(self, n_epochs):
        self.test_epoch(0)
        for epoch in range(1, n_epochs + 1):
            self.train_epoch(epoch)
            self.test_epoch(epoch)


if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("mnist_add_two_numbers_sampling")
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size-train", type=int, default=64)
    parser.add_argument("--batch-size-test", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--difficulty", type=str, default="easy")
    args = parser.parse_args()

    # Read json
    dir_path = os.path.dirname(os.path.realpath(__file__))
    configuration = json.load(
        open(os.path.join(dir_path, "configuration.json")))

    # Parameters
    n_epochs = args.n_epochs
    batch_size_train = args.batch_size_train
    batch_size_test = args.batch_size_test
    learning_rate = args.learning_rate
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Dataloaders
    for task in configuration:
        task_config = configuration[task]
        train_loader, test_loader = train_test_loader(
            task_config, batch_size_train, batch_size_test)

        py_func = task_config[PY_PROGRAM]
        unstructured_datasets = [task_dataset.TaskDataset.get_unstructured_dataset(
            input, train=True) for input in task_config[INPUTS]]
        fn = task_program.dispatcher[py_func]

        # Create trainer and train
        trainer = Trainer(train_loader, test_loader,
                          learning_rate, unstructured_datasets, fn)
        trainer.train(n_epochs)
