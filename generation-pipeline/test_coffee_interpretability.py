import run_benchmarks

from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Pool
import os
import csv
import random
import json
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from constants import *
import unstructured_dataset
import structured_dataset
import task_dataset
import task_program
import output
import blackbox


def train_test_loader(configuration, batch_size_train, batch_size_test):
    train_loader = torch.utils.data.DataLoader([])

    test_loader = torch.utils.data.DataLoader(
        run_benchmarks.Dataset(configuration, train=False),
        collate_fn=run_benchmarks.Dataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=True
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("neuro-symbolic-dataset")
    parser.add_argument("--n-epochs", type=int, default=30)
    # parser.add_argument("--seed", type=int, default=1234)
    # parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--configuration", type=str,
                        default="configuration.json")
    parser.add_argument("--symmetry", type=bool, default=False)
    parser.add_argument("--caching", type=bool, default=True)
    parser.add_argument("--threaded", type=int, default=0)
    args = parser.parse_args()

    # random_seeds = [3177, 5848, 9175]
    random_seeds = [9175]
    sample_counts = [100]
    tasks = ['miner_coffee_leaf_severity']

    accuracies = ["accuracy epoch " + str(i+1) for i in range(10)]
    times = ["time epoch " + str(i+1) for i in range(10)]
    field_names = ['task name', 'random seed',
                   'sample count'] + accuracies + times

    with open('coffee_baseline.csv', 'w', newline='') as csvfile:
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
                output_config = task_config[OUTPUT]
                om = output.get_output_mapping(output_config)

                # Create trainer and train
                py_func = task_config[PY_PROGRAM]
                learning_rate = task_config[LEARNING_RATE]
                fn = task_program.dispatcher[py_func]
                config = task_config[INPUTS]
                unstructured_datasets = [task_dataset.TaskDataset.get_unstructured_dataset(
                    input, train=True) for input in task_config[INPUTS]]
                trainer = run_benchmarks.Trainer(train_loader=train_loader,
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

                trainer.network.nets[0].load_state_dict(pickle.load(
                    open("model/miner_coffee_leaf_severity/9175-13.pkl", "rb")))

                # dict = trainer.test_epoch(0)

                with torch.no_grad():
                    iter = tqdm(trainer.test_loader,
                                total=len(trainer.test_loader))
                    for i, (data, target) in enumerate(iter):
                        print(data["x"][0].shape)
                        output_tensor = trainer.network.nets[0](data["x"][0])
                        print(output_tensor)
                        print(output_tensor.shape)
                        for j in range(16):
                            for k in range(46):
                                img = data["x"][0][j][k]
                                reshaped_image = img.transpose(0, 2)
                                plt.imshow(reshaped_image)
                                plt.axis('off')

                                dir = f"model/miner_coffee_leaf_severity/pred/{i}-{j}/"
                                if not os.path.exists(dir):
                                    os.makedirs(dir, exist_ok=True)
                                argmax = torch.argmax(
                                    output_tensor[j][k], dim=0).item()
                                pred = "pos" if argmax == 1 else "neg"
                                prob = output_tensor[j][k][argmax]
                                plt.savefig(os.path.join(
                                    dir, f"{k}-pred-{pred}-{prob:.4f}.jpg"), bbox_inches='tight', pad_inches=0)
