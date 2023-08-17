
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
from unstructured import COFFEE_dataset
from unstructured import COFFEE_net
from unstructured import CoNLL2003_dataset
from unstructured import CoNLL2003_net


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
        return MNIST_net.MNISTNet(n_preds=10)

    def confusion_matrix(self, network):
        digits = [i for i in range(10)]
        mnist_dataset, _ = MNIST_dataset.get_data(train=False, digits=digits)
        self.plot_confusion_matrix(network=network, dataset=mnist_dataset)


class MNISTDataset_0to4(UnstructuredDataset):

    def __init__(self, train):
        self.name = MNIST_0TO4
        self.digits = [i for i in range(0, 5)]
        self.data, self.ids_of_digit = MNIST_dataset.get_data(
            train=train, digits=self.digits)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        return MNIST_dataset.MNISTDataset.collate_fn(batch)

    def input_mapping(self):
        return [i for i in range(1, 5)]

    def sample_with_y(self, digit: int) -> int:
        return self.ids_of_digit[digit][random.randrange(0, len(self.ids_of_digit[digit]))]

    def get(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.data[index]

    def net(self):
        return MNIST_net.MNISTNet(n_preds=4)

    def confusion_matrix(self, network):
        mnist_dataset, _ = MNIST_dataset.get_data(
            train=False, digits=self.digits)
        self.plot_confusion_matrix(network=network, dataset=mnist_dataset)


class EMNISTDataset(UnstructuredDataset):
    def __init__(self, train):
        self.name = EMNIST
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
        return SVHN_net.SVHNNet()

    def confusion_matrix(self, network):
        svhn_dataset, _ = SVHN_dataset.get_data(train=False)
        self.plot_confusion_matrix(network=network, dataset=svhn_dataset)


class HWFDataset(UnstructuredDataset):

    def __init__(self, train):
        self.name = HWF_SYMBOL
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
        self.name = MNIST_VIDEO
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


class CoffeeLeafMinerDataset(UnstructuredDataset):

    def __init__(self, train):
        self.name = COFFEE_LEAF_MINER
        self.data, self.ids_of_severity = COFFEE_dataset.get_data(
            prefix='miner', train=train)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        return COFFEE_dataset.collate_fn(batch)

    def input_mapping(self):
        return [i for i in range(1, 6)]

    def sample_with_y(self, severity):
        return self.ids_of_severity[severity + 1][random.randrange(0, len(self.ids_of_severity[severity + 1]))]

    def get(self, index: int) -> Tuple[Tuple[List[torch.Tensor], List[int]], int]:
        return self.data[index]

    def net(self):
        return COFFEE_net.COFFEE_net()

    def confusion_matrix(self, network):
        coffee_dataset, _ = COFFEE_dataset.get_data(
            prefix='miner', train=False)
        self.plot_confusion_matrix(network=network, dataset=coffee_dataset)


class CoffeeLeafRustDataset(UnstructuredDataset):

    def __init__(self, train):
        self.name = COFFEE_LEAF_RUST
        self.data, self.ids_of_severity = COFFEE_dataset.get_data(
            prefix='rust', train=train)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        return COFFEE_dataset.collate_fn(batch)

    def input_mapping(self):
        return [i for i in range(1, 6)]

    def sample_with_y(self, severity):
        return self.ids_of_severity[severity + 1][random.randrange(0, len(self.ids_of_severity[severity + 1]))]

    def get(self, index: int) -> Tuple[Tuple[List[torch.Tensor], List[int]], int]:
        return self.data[index]

    def net(self):
        return COFFEE_net.COFFEE_net()

    def confusion_matrix(self, network):
        coffee_dataset, _ = COFFEE_dataset.get_data(prefix='rust', train=False)
        self.plot_confusion_matrix(network=network, dataset=coffee_dataset)


class CoNLL2003Dataset(UnstructuredDataset):

    def __init__(self, train):
        self.name = CONLL2003
        self.data = CoNLL2003Dataset.get_data(train=train)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        pass

    def input_mapping(self):
        return [i for i in range(0)]

    def sample_with_y(self, sequence_id: int) -> int:
        return sequence_id

    def get(self, index: int):  # -> Tuple[torch.Tensor, int]:
        return self.data[index]

    def net(self):
        return CoNLL2003_net.BertModel(self.data.unique_labels)

    def confusion_matrix(self, network):
        pass
