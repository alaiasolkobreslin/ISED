
from typing import *
import random
import torchvision

from sklearn.metrics import confusion_matrix
import numpy

import torch

from constants import *
from unstructured import MNIST_dataset
from unstructured import MNIST_net
from unstructured import EMNIST_dataset
from unstructured import EMNIST_net
from unstructured import SVHN_dataset
from unstructured import SVHN_net
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

    def confusion_matrix(self, network):
        mnist_dataset, _ = MNIST_dataset.get_data(train=False)
        self.plot_confusion_matrix(network=network, dataset=mnist_dataset)


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

    def confusion_matrix(self, network):
        emnist_dataset, _ = EMNIST_dataset.get_data(train=False)
        self.plot_confusion_matrix(network=network, dataset=emnist_dataset)


class SVHNDataset(UnstructuredDataset):
    def __init__(self, train):
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
        return SVHN_net.SVHNNet()

    def confusion_matrix(self, network):
        svhn_dataset, _ = SVHN_dataset.get_data(train=False)
        self.plot_confusion_matrix(network=network, dataset=svhn_dataset)


class HWFDataset(UnstructuredDataset):

    def __init__(self, train):
        self.data, self.ids_of_expr = HWF_dataset.get_data(train)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        return HWF_dataset.HWFDataset.collate_fn(batch)

    def input_mapping(self):
        return [str(x) for x in range(10)] + ['+', '-', '*', '/']

    def sample_with_y(self, expr_id: int) -> int:
        expr = self.data.metadata[expr_id]['expr']
        return self.ids_of_expr[expr][random.randrange(0, len(self.ids_of_expr[expr]))]

    def get(self, index: int) -> Tuple[torch.Tensor, str]:
        (expr, string, _) = self.data[index]
        return ((expr, len(string)), string)

    def net(self):
        return HWF_symbol_net.SymbolNet()

    def confusion_matrix(self, network):
        pass


class MNISTVideoDataset(UnstructuredDataset):

    def __init__(self, train):
        self.data, self.ids_of_video = MNIST_video_dataset.get_data(train)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        return MNIST_video_dataset.MNISTVideoDataset.collate_fn(batch)

    def input_mapping(self):
        return [i for i in range(10)]

    def sample_with_y(self, video_id):
        video = self.data[video_id]
        key = tuple(video[1][0])
        return self.ids_of_video[key][random.randrange(0, len(self.ids_of_video[key]))]

    def get(self, index: int) -> Tuple[List[torch.Tensor], Tuple[Any]]:
        imgs = self.data[index][0]
        _, digits_tensor, change_tensor = self.data[index][1]
        return (imgs, (digits_tensor, change_tensor))

    def net(self):
        return MNIST_video_net.MNISTVideoCNN()

    def confusion_matrix(self, network):
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

    def confusion_matrix(self, network):
        pass
