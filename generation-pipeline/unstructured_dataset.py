
from typing import *
import random

import torch

from constants import *
from unstructured import MNIST_dataset
from unstructured import MNIST_net
from unstructured import EMNIST_dataset
from unstructured import EMNIST_net
from unstructured import HWF_dataset
from unstructured import HWF_symbol_net
from unstructured import MNIST_video_dataset
from unstructured import MNIST_video_net


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


class MNISTDataset(UnstructuredDataset):

    def __init__(self, train):
        self.data, self.ids_of_digit = MNIST_dataset.get_data(
            train)

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
        return MNIST_net.MNISTNet()


class EMNISTDataset(UnstructuredDataset):
    def __init__(self, train):
        self.data, self.ids_of_character = EMNIST_dataset.get_data(
            train)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        return EMNIST_dataset.EMNISTDataset.collate_fn(batch)

    def input_mapping(self):
        return [i for i in range(47)]

    def sample_with_y(self, character: int) -> int:
        return self.ids_of_character[character][random.randrange(0, len(self.ids_of_character[character]))]

    def get(self, index: int) -> Tuple[torch.Tensor, chr]:
        return self.data[index]

    def net(self):
        return EMNIST_net.EMNISTNet()


class HWFDataset(UnstructuredDataset):

    def __init__(self, train):
        self.data, self.ids_of_expr = HWF_dataset.get_data(train)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        return HWF_dataset.HWFDataset.collate_fn(batch)

    def input_mapping(self):
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/']

    def sample_with_y(self, expr_id: int) -> int:
        expr = self.data.metadata[expr_id]['expr']
        return self.ids_of_expr[expr][random.randrange(0, len(self.ids_of_expr[expr]))]

    def get(self, index: int) -> Tuple[torch.Tensor, str]:
        (expr, string, _) = self.data[index]
        return ((expr, len(string)), string)

    def net(self):
        return HWF_symbol_net.SymbolNet()


class MNISTVideoDataset(UnstructuredDataset):

    def __init__(self, train):
        pass

    def __len__(self):
        pass

    @staticmethod
    def collate_fn(batch):
        pass

    def input_mapping(self):
        pass

    def sample_with_y(self):
        pass

    def get(self):
        pass

    def net(self):
        pass


class MNISTGridDataset(UnstructuredDataset):

    def __init__(self, train):
        pass

    def __len__(self):
        pass

    @staticmethod
    def collate_fn(batch):
        pass

    def input_mapping(self):
        pass

    def sample_with_y(self):
        pass

    def get(self):
        pass

    def net(self):
        pass
