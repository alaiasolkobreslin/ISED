
from typing import *
import random

import torch

import MNIST_dataset
import MNIST_net


class UnstructuredDataset:

    def sample_with_y(self, y):
        pass

    def get(self, id):
        pass

    def net(self):
        pass


class MNISTDataset(UnstructuredDataset):

    def __init__(self, train):
        self.data, self.ids_of_digit = MNIST_dataset.get_data(train)

    def sample_with_y(self, digit: int) -> int:
        return self.ids_of_digit[digit][random.randrange(0, len(self.ids_of_digit[digit]))]

    def get(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.data[index]

    def net(self):
        return MNIST_net.MNISTNet()


class MNISTVideoDataset(UnstructuredDataset):
    def __init__(self):
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

    def sample_with_y(self):
        pass

    def get(self):
        pass

    def net(self):
        pass
