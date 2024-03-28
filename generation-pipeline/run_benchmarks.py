import os
import pickle
import json
import csv
import time
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
import math

import unstructured_dataset
import structured_dataset
import task_dataset
import task_program
import output
import blackbox
from constants import *


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            config: dict,
            train: bool):
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
        imgs = {input[NAME]: collate_fns[input[NAME]](
            [data_dict[input[NAME]] for data_dict in data_dicts], input) for input in config}
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
    def __init__(
            self,
            unstructured_datasets: List[unstructured_dataset.UnstructuredDataset],
            config: dict,
            fn: Callable,
            output_mapping: output.OutputMapping,
            sample_count: int,
            batch_size_train: int,
            check_symmetry: bool,
            caching: bool):
        super(TaskNet, self).__init__()

        self.config = config
        self.unstructured_datasets = unstructured_datasets
        self.structured_datasets = [
            structured_dataset.get_structured_dataset_static(input) for input in config]

        self.nets_dict = {}
        self.nets = self.get_nets_list()

        self.forward_fns = [partial(sd.forward, self.nets[i])
                            for i, sd in enumerate(self.structured_datasets)]
        input_mappings = tuple([sd.get_input_mapping(
            config[i]) for i, sd in enumerate(self.structured_datasets)])
        loss_aggregator = task_config.get(LOSS_AGGREGATOR, ADD_MULT)
        self.eval_formula = \
            blackbox.BlackBoxFunction(function=fn,
                                      input_mappings=input_mappings,
                                      output_mapping=output_mapping,
                                      batch_size=batch_size_train,
                                      loss_aggregator=loss_aggregator,
                                      check_symmetry=check_symmetry,
                                      caching=caching,
                                      sample_count=sample_count)

        self.pool = Pool(processes=batch_size_train)

    def get_nets_list(self):
        nets = []

        def add_net(ud_name, ud):
            if ud_name not in self.nets_dict:
                self.nets_dict[ud_name] = ud.net()
            nets.append(self.nets_dict[ud_name])
        for ud in self.unstructured_datasets:
            add_net(ud.name, ud)
        return nets

    def parameters(self):
        return [net.parameters() for net in self.nets_dict.values()]

    def task_test(self, args, x):
        return self.sampling.sample_test(args, data=x)

    def forward(self, x):
        keys = [key for key in x]
        distrs = [self.forward_fns[i](x[key]) for i, key in enumerate(keys)]
        inputs = [self.structured_datasets[i].distrs_to_input(distrs[i], x[keys[i]], input)
                  for i, input in enumerate(self.config)]
        return self.eval_formula(*inputs)

    def eval(self):
        for net in self.nets_dict.values():
            net.eval()

    def train(self):
        for net in self.nets_dict.values():
            net.train()

    def close(self):
        self.pool.close()

    def confusion_matrix(self):
        # for i, ud in enumerate(self.unstructured_datasets):
        #     ud.confusion_matrix(self.nets[i])
        self.unstructured_datasets[0].confusion_matrix(self.nets[0])


class Trainer():
    def __init__(
            self,
            train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader,
            unstructured_datasets: List[unstructured_dataset.UnstructuredDataset],
            learning_rate: float,
            config: dict,
            fn: Callable,
            output_mapping: output.OutputMapping,
            sample_count: int,
            batch_size_train: int,
            check_symmetry: bool,
            caching: bool):
        self.network = TaskNet(unstructured_datasets=unstructured_datasets,
                               config=config,
                               fn=fn,
                               output_mapping=output_mapping,
                               sample_count=sample_count,
                               batch_size_train=batch_size_train,
                               check_symmetry=check_symmetry,
                               caching=caching)
        self.output_mapping = output_mapping
        self.optimizers = [optim.Adam(
            net.parameters(), lr=learning_rate) for net in self.network.nets_dict.values()]
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = F.binary_cross_entropy

    def train_epoch(self, epoch):
        self.network.train()
        num_items = 0
        train_loss = 0
        total_correct = 0
        iter = tqdm(self.train_loader, total=len(self.train_loader))
        for (i, (data, target)) in enumerate(iter):
            (output_mapping, y_pred_sim, y_pred) = self.network(data)

            # Normalize label format
            # batch_size, num_outputs = y_pred.shape
            batch_size = y_pred_sim.shape[0]
            num_outputs = 1
            norm_label, y = self.output_mapping.get_normalized_labels(
                y_pred_sim, target, output_mapping)

            if output_mapping:
                # Compute loss
                loss = self.loss_fn(y_pred_sim, norm_label)
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
                loss.backward()
                for optimizer in self.optimizers:
                    optimizer.step()
                if not math.isnan(loss.item()):
                    train_loss += loss.item()

            # Collect index and compute accuracy
            if output_mapping:
                y_index = torch.argmax(y, dim=1)
                y_pred_index = torch.argmax(y_pred, dim=1)
                correct_count = torch.sum(torch.where(torch.sum(
                    y, dim=1) > 0, y_index == y_pred_index, torch.zeros(batch_size, device=DEVICE).bool())).item()
            else:
                correct_count = 0

            # Stats
            num_items += batch_size
            total_correct += correct_count
            perc = 100. * total_correct / num_items
            avg_loss = train_loss / (i + 1)

            # Prints
            iter.set_description(
                f"[Train Epoch {epoch}] Avg loss: {avg_loss:.4f}, Accuracy: {total_correct}/{num_items} ({perc:.2f}%)")

    def test_epoch(self, epoch):
        self.network.eval()
        num_items = 0
        test_loss = 0
        total_correct = 0
        with torch.no_grad():
            iter = tqdm(self.test_loader, total=len(self.test_loader))
            for i, (data, target) in enumerate(iter):
                (output_mapping, y_pred_sim, y_pred) = self.network(data)

                # Normalize label format
                # batch_size, num_outputs = y_pred.shape
                batch_size = y_pred_sim.shape[0]
                num_outputs = 1

                norm_label, y = self.output_mapping.get_normalized_labels(
                    y_pred_sim, target, output_mapping)

                # Compute loss
                loss = self.loss_fn(y_pred_sim, norm_label)
                if not math.isnan(loss.item()):
                    test_loss += loss.item()

                # Collect index and compute accuracy
                if output_mapping:
                    y_index = torch.argmax(y, dim=1)
                    y_pred_index = torch.argmax(
                        y_pred, dim=1)
                    correct_count = torch.sum(torch.where(torch.sum(
                        y, dim=1) > 0, y_index == y_pred_index, torch.zeros(batch_size, device=DEVICE).bool())).item()
                else:
                    correct_count = 0

                # Stats
                num_items += batch_size
                total_correct += correct_count
                perc = 100. * total_correct / num_items
                avg_loss = test_loss / (i + 1)

                # Prints
                iter.set_description(
                    f"[Test Epoch {epoch}] Avg loss: {avg_loss:.4f}, Accuracy: {total_correct}/{num_items} ({perc:.2f}%)")

        dir = f"model/{task}/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        pickle.dump(self.network.nets[0].state_dict(), open(
            os.path.join(dir, f"{seed}-{epoch}.pkl"), "wb"))

        return total_correct / num_items
        # self.network.confusion_matrix()

    def train(self, n_epochs):
        dict = {}
        # self.test_epoch(0)
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            self.train_epoch(epoch)
            t1 = time.time()
            dict["time epoch " + str(epoch)] = round(t1 - t0, ndigits=4)
            acc = self.test_epoch(epoch)
            dict["accuracy epoch " + str(epoch)] = round(acc, ndigits=6)
        self.network.close()
        return dict


if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("neuro-symbolic-dataset")
    parser.add_argument("--n-epochs", type=int, default=10)
    # parser.add_argument("--seed", type=int, default=1234)
    # parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--configuration", type=str,
                        default="configuration.json")
    parser.add_argument("--symmetry", type=bool, default=False)
    parser.add_argument("--caching", type=bool, default=True)
    parser.add_argument("--threaded", type=int, default=0)
    args = parser.parse_args()

    random_seeds = [3177, 5848]
    sample_counts = [1000]
    tasks = ['sort_list_indices_length9_mnist', 'sort_list_indices_length10_mnist']

    accuracies = ["accuracy epoch " + str(i+1) for i in range(args.n_epochs)]
    times = ["time epoch " + str(i+1) for i in range(args.n_epochs)]
    field_names = ['task name', 'random seed',
                   'sample count'] + accuracies + times

    with open('sort_9_10.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        csvfile.close()

    # environment init
    torch.multiprocessing.set_start_method('spawn')

    # Read json
    dir_path = os.path.dirname(os.path.realpath(__file__))
    configuration = json.load(
        open(os.path.join(dir_path, args.configuration)))

    # Parameters
    n_epochs = args.n_epochs

    for task in tasks:
        for n_samples in sample_counts:
            for seed in random_seeds:
                torch.manual_seed(seed)
                random.seed(seed)
                print('Task: {}'.format(task))

                task_config = configuration[task]

                # Initialize the train and test loaders
                batch_size_train = task_config[BATCH_SIZE_TRAIN]
                batch_size_test = task_config[BATCH_SIZE_TEST]
                train_loader, test_loader = train_test_loader(
                    task_config, batch_size_train, batch_size_test)

                # Set the output mapping
                om = output.get_output_mapping(task_config)

                # Create trainer and train
                py_func = task_config[PY_PROGRAM]
                learning_rate = task_config[LEARNING_RATE]
                fn = task_program.dispatcher[py_func]
                config = task_config[INPUTS]
                unstructured_datasets = [task_dataset.TaskDataset.get_unstructured_dataset(
                    input, train=True) for input in task_config[INPUTS]]
                trainer = Trainer(train_loader=train_loader,
                                  test_loader=test_loader,
                                  unstructured_datasets=unstructured_datasets,
                                  learning_rate=learning_rate,
                                  config=config,
                                  fn=fn,
                                  output_mapping=om,
                                  sample_count=n_samples,
                                  batch_size_train=batch_size_train,
                                  check_symmetry=args.symmetry,
                                  caching=args.caching)
                dict = trainer.train(n_epochs)
                dict["task name"] = task
                dict["random seed"] = seed
                dict["sample count"] = n_samples
                with open('sample_count_experiment23.csv', 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=field_names)
                    writer.writerow(dict)
                    csvfile.close()
