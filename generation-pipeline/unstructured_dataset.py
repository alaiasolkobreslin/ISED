
from typing import *
import random
import string

from sklearn.metrics import confusion_matrix
import numpy

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

    def confusion_matrix(self, network):
        """
        Plots the confusion matrix
        """
        pass

    def plot_confusion_matrix(self, network, dataset):
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=64)

        network.eval()

        y_true, y_pred = [], []
        with torch.no_grad():
            for (imgs, gt) in loader:
                preds = numpy.argmax(network(imgs), axis=1)
                y_true += [d.item() for d in gt]
                y_pred += [d.item() for d in preds]

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        print(cm)

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

    def confusion_matrix(self, network):
        digits = [i for i in range(10)]
        mnist_dataset, _ = MNIST_dataset.get_data(train=False, digits=digits)
        self.plot_confusion_matrix(network=network, dataset=mnist_dataset)


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

    def confusion_matrix(self, network):
        svhn_dataset, _ = SVHN_dataset.get_data(train=False)
        self.plot_confusion_matrix(network=network, dataset=svhn_dataset)

