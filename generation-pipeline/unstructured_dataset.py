
from typing import *
import random

import torch

from constants import *
from unstructured import MNIST_dataset
from unstructured import MNIST_net
from unstructured import SVHN_dataset
from unstructured import SVHN_net


class UnstructuredDataset:

    def __len__(self):
        pass

    def collate_fn(batch):
        pass

    def input_mapping(self):
        pass

    def sample_with_y(self, y):
        """
        Returns a random datapoint from the unstructured dataset in which the ground truth is `y`
        """
        pass

    def get(self, id):
        """
        Returns the datapoint at index `id` for this unstructured dataset
        """
        pass

    def net(self):
        """
        Returns a neural network for this unstructured dataset
        """
        pass

    def get_full_dataset(self):
        """
        Returns the entire unstructured dataset
        """
        pass


class MNISTDataset(UnstructuredDataset):

    def __init__(self, train):
        self.name = MNIST
        digits = [i for i in range(10)]
        self.data, self.ids_of_digit = MNIST_dataset.get_data(
            train=train, digits=digits)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        return MNIST_dataset.MNISTDataset.collate_fn(batch)

    def input_mapping(self):
        return [i for i in range(10)]

    def sample_with_y(self, digit: int) -> int:
        return self.ids_of_digit[digit][random.randrange(0, len(self.ids_of_digit[digit]))]

    def get(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.data[index]

    def net(self):
        return MNIST_net.MNISTNet(n_preds=10).to(DEVICE)


class SVHNDataset(UnstructuredDataset):
    def __init__(self, train):
        self.name = SVHN
        self.data, self.ids_of_digit = SVHN_dataset.get_data(
            train)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        return SVHN_dataset.SVHNDataset.collate_fn(batch)

    def input_mapping(self):
        return [i for i in range(10)]

    def sample_with_y(self, digit: int) -> int:
        return self.ids_of_digit[digit][random.randrange(0, len(self.ids_of_digit[digit]))]

    def get(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.data[index]

    def net(self):
        return SVHN_net.SVHNNet().to(DEVICE)
