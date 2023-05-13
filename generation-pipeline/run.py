import os
import json
import random
from typing import *
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Pool

from argparse import ArgumentParser
from tqdm import tqdm

import unstructured_dataset
import structured_dataset
import task_dataset
import task_program
import sample
from constants import *


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, train):
        self.dataset = task_dataset.TaskDataset(config, train)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    @staticmethod
    def collate_fn(batch):
        data_dicts = [item[0] for item in batch]
        config = batch[0][1]
        collate_fns = {input[NAME]: structured_dataset.get_structured_dataset_static(
            input).collate_fn for input in config}
        imgs = {key: collate_fns[key](
            [data_dict[key] for data_dict in data_dicts]) for key in data_dicts[0].keys()}
        results = [item[2] for item in batch]
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
    def __init__(self, unstructured_datasets, config, fn):
        super(TaskNet, self).__init__()

        self.nets_dict = {}
        self.set_nets_list()

        n_inputs = len(self.nets)
        structured_datasets = [
            structured_dataset.get_structured_dataset_static(input) for input in config]
        self.flatten_fns = [partial(sd.flatten, config[i])
                            for i, sd in enumerate(structured_datasets)]
        unflatten_fns = [(partial(sd.unflatten, config[i]), sd.n_unflatten(config[i]))
                         for i, sd in enumerate(structured_datasets)]
        self.forward_fns = [partial(sd.forward, self.nets[i])
                            for i, sd in enumerate(structured_datasets)]

        self.sampling = sample.StandardSample(
            n_inputs, args.n_samples, fn, self.flatten_fns, unflatten_fns, args.threaded)
        self.sampling_fn = self.sampling.sample_train_backward_threaded if args.threaded else self.sampling.sample_train_backward

        self.pool = Pool(processes=args.batch_size_train)

    def set_nets_list(self):
        self.nets = []
        for ud in unstructured_datasets:
            if type(ud) is unstructured_dataset.MNISTDataset:
                if MNIST not in self.nets_dict:
                    self.nets_dict[MNIST] = ud.net()
                self.nets.append(self.nets_dict[MNIST])
            elif type(ud) is unstructured_dataset.EMNISTDataset:
                if EMNIST not in self.nets_dict:
                    self.nets_dict[EMNIST] = ud.net()
                self.nets.append(self.nets_dict[EMNIST])
            elif type(ud) is unstructured_dataset.HWFDataset:
                if HWF_SYMBOL not in self.nets_dict:
                    self.nets_dict[HWF_SYMBOL] = ud.net()
                self.nets.append(self.nets_dict[HWF_SYMBOL])

    def parameters(self):
        return [net.parameters() for net in self.nets_dict.values()]

    def task_test(self, args, x):
        return self.sampling.sample_test(args, data=x)

    def forward(self, x, y):
        distrs = [self.forward_fns[i](x[key]) for i, key in enumerate(x)]
        flattened = []
        for i, distr in enumerate(distrs):
            flattened += self.flatten_fns[i](distr)
        distrs = flattened
        distrs_detached = [distr.detach() for distr in distrs]
        batch_nums = [i for i in range(len(y))]
        argss = list(zip(*(tuple(distrs_detached)), y, batch_nums))
        out_pred = self.pool.map(
            partial(self.sampling_fn, [val for val in x.values()]), argss)
        out_pred = list(zip(*out_pred))
        grads = [torch.stack(grad) for grad in out_pred]

        for i in range(len(grads)):
            distrs[i].backward(grads[i], retain_graph=True)

        return abs(torch.mean(torch.cat(tuple(grads))))

    def evaluate(self, x):
        """
        Invoked during testing
        """
        distrs = [self.forward_fns[i](x[key]) for i, key in enumerate(x)]
        return self.task_test(distrs, [val for val in x.values()])

    def eval(self):
        for net in self.nets_dict.values():
            net.eval()

    def train(self):
        for net in self.nets_dict.values():
            net.train()


class Trainer():
    def __init__(self, train_loader, test_loader, learning_rate, unstructured_datasets, config, fn):
        self.network = TaskNet(unstructured_datasets, config, fn)
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
                    if output[i] == target[i]:
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
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--batch-size-train", type=int, default=64)
    parser.add_argument("--batch-size-test", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--difficulty", type=str, default="easy")
    parser.add_argument("--threaded", type=int, default=0)
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
        print('Task: {}'.format(task))
        task_config = configuration[task]
        train_loader, test_loader = train_test_loader(
            task_config, batch_size_train, batch_size_test)

        py_func = task_config[PY_PROGRAM]
        unstructured_datasets = [task_dataset.TaskDataset.get_unstructured_dataset(
            input, train=True) for input in task_config[INPUTS]]
        structured_datasets = [task_dataset.TaskDataset.get_structured_dataset(
            task_config[INPUTS][i], ud) for i, ud in enumerate(unstructured_datasets)]
        fn = task_program.dispatcher[py_func]

        # Create trainer and train
        trainer = Trainer(train_loader, test_loader,
                          learning_rate, unstructured_datasets, task_config[INPUTS], fn)
        trainer.train(n_epochs)
