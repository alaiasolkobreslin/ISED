
from typing import *
import random

import torch

import MNIST_dataset

class UnstructuredDataset:
    
    def sample_with_y(self, y):
        pass
    
    def get(self, id):
        pass

class MNISTDataset(UnstructuredDataset):
    
    def __init__(self):
        self.data, self.ids_of_digit = MNIST_dataset.get_data()

    def sample_with_y(self, digit: int) -> int:
        return self.ids_of_digit[digit][random.randrange(0, len(self.ids_of_digit[digit]))]
    
    def get(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.data[index]
    
class MNISTVideoDataset(UnstructuredDataset):
    def __init__(self):
        pass

    def sample_with_y(self):
        pass

    def get(self):
        pass

class MNISTGridDataset(UnstructuredDataset):
    def __init__(self):
        pass

    def sample_with_y(self):
        pass

    def get(self):
        pass
