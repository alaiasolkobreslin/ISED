
from typing import *
import random

import torch

from unstructured import MNIST_dataset
from unstructured import MNIST_net
from unstructured import HWF_dataset
from unstructured import HWF_symbol_net


class UnstructuredDataset:

    def __len__(self):
        pass

    def collate_fn(self, batch):
        pass

    def sample_with_y(self, y):
        pass

    def get(self, id):
        pass

    def net(self):
        pass


class MNISTDataset(UnstructuredDataset):

    def __init__(self, train):
        self.data, self.ids_of_digit = MNIST_dataset.get_data(train)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        return MNIST_dataset.MNISTDataset.collate_fn(batch)

    def sample_with_y(self, digit: int) -> int:
        return self.ids_of_digit[digit][random.randrange(0, len(self.ids_of_digit[digit]))]

    def get(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.data[index]

    def net(self):
        return MNIST_net.MNISTNet()


class HWFDataset(UnstructuredDataset):

    def __init__(self, train):
        self.data, self.ids_of_expr = HWF_dataset.get_data(train)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        return HWF_dataset.HWFDataset.collate_fn(batch)

    def sample_with_y(self, expr_id: int) -> int:
        expr = self.data.metadata[expr_id]['expr']
        return self.ids_of_expr[expr][random.randrange(0, len(self.ids_of_expr[expr]))]

    def get(self, index: int) -> Tuple[torch.Tensor, str]:
        return self.data[index]

    def net(self):
        return HWF_symbol_net.SymbolNet()


class MNISTVideoDataset(UnstructuredDataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def sample_with_y(self):
        pass

    def get(self):
        pass

    def net(self):
        pass


class MNISTGridDataset(UnstructuredDataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def sample_with_y(self):
        pass

    def get(self):
        pass

    def net(self):
        pass
