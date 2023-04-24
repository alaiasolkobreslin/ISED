import torch

from constants import *
import strategy
import unstructured_dataset


class StructuredDataset:

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def collate_fn(batch):
        pass

    def forward(net, x):
        pass

    def generate_datapoint(self):
        pass

    def get_strategy(self):
        pass

    def generate_dataset(self):
        pass

    def flatten(config, input):
        pass

    def unflatten(config, samples):
        pass

    def n_unflatten(config):
        pass


class SingleIntDataset(StructuredDataset):

    def __init__(self, config, unstructured_dataset):
        self.config = config
        self.unstructured_dataset = unstructured_dataset
        self.strategy = self.get_strategy()
        self.dataset = self.generate_dataset()

    def __len__(self):
        return len(self.unstructured_dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch)

    def forward(net, x):
        return net(x)

    def generate_datapoint(self):
        return self.strategy.sample()

    def get_strategy(self):
        s = self.config[STRATEGY]
        input_mapping = [i for i in range(10)]
        if s == SINGLETON_STRATEGY:
            strat = strategy.SingletonStrategy(
                self.unstructured_dataset, input_mapping)
        return strat

    def generate_dataset(self):
        length = self.__len__()
        dataset = [None] * length
        for i in range(length):
            dataset[i] = self.generate_datapoint()

    def flatten(config, input):
        return [input]

    def unflatten(config, samples):
        return samples

    def n_unflatten(config):
        return 1


class IntDataset(StructuredDataset):

    def __init__(self, config, unstructured_dataset):
        self.config = config
        self.unstructured_dataset = unstructured_dataset
        self.strategy = self.get_strategy()
        self.dataset = self.generate_dataset()

    def __len__(self):
        return int(len(self.unstructured_dataset) / self.config[N_DIGITS])

    def __getitem__(self, index):
        return self.dataset[index]

    @staticmethod
    def collate_fn(batch):
        imgs = [torch.stack([item[i] for item in batch])
                for i in range(len(batch[0]))]
        return imgs

    def forward(net, x):
        return [net(item) for item in x]

    def get_strategy(self):
        n_digits = self.config[N_DIGITS]
        s = self.config[STRATEGY]
        input_mapping = [i for i in range(10)]
        if s == SIMPLE_LIST_STRATEGY:
            strat = strategy.SimpleListStrategy(
                self.unstructured_dataset, input_mapping, n_digits)
        return strat

    def generate_datapoint(self):
        samples = self.strategy.sample()
        imgs, number_lst = zip(*samples)
        number = ''.join(str(n) for n in number_lst)
        return (imgs, int(number))

    def generate_dataset(self):
        length = self.__len__()
        dataset = [None] * length
        for i in range(length):
            dataset[i] = self.generate_datapoint()

    def flatten(config, input):
        return input

    def unflatten(config, samples):
        number = ''
        for i in samples:
            number += str(i.item())
        return [torch.tensor(int(number))]

    def n_unflatten(config):
        return config[N_DIGITS]


class SingleIntListDataset(StructuredDataset):
    def __init__(self, config, unstructured_dataset):
        self.config = config
        self.unstructured_dataset = unstructured_dataset
        self.strategy = self.get_strategy()
        self.dataset = self.generate_dataset()

    def __len__(self):
        return int(len(self.unstructured_dataset) / self.config[LENGTH])

    def __getitem__(self, index):
        return self.dataset[index]

    @staticmethod
    def collate_fn(batch):
        imgs = [torch.stack([item[i] for item in batch])
                for i in range(len(batch[0]))]
        return imgs

    def forward(net, x):
        return [net(item) for item in x]

    def get_strategy(self):
        length = self.config[LENGTH]
        s = self.config[STRATEGY]
        input_mapping = [i for i in range(10)]
        if s == SIMPLE_LIST_STRATEGY:
            strat = strategy.SimpleListStrategy(
                self.unstructured_dataset, input_mapping, length)
        return strat

    def generate_datapoint(self):
        samples = self.strategy.sample()
        return zip(*samples)
        # return self.strategy.sample()

    def generate_dataset(self):
        length = self.__len__()
        dataset = [None] * length
        for i in range(length):
            dataset[i] = self.generate_datapoint()

    def flatten(config, input):
        return input

    def unflatten(config, samples):
        return [samples]

    def n_unflatten(config):
        return config[LENGTH]


class IntListDataset(StructuredDataset):

    def __init__(self, config, unstructured_dataset):
        self.config = config
        self.unstructured_dataset = unstructured_dataset
        self.strategy = self.get_strategy()
        self.dataset = self.generate_dataset()

    def __len__(self):
        return int(len(self.unstructured_dataset) / (self.config[N_DIGITS] * self.config[LENGTH]))

    def __getitem__(self, index):
        return self.dataset[index]

    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch)

    def forward(net, x):
        return [[net(i) for i in item] for item in x]

    def get_strategy(self):
        n_digits = self.config[N_DIGITS]
        s = self.config[STRATEGY]
        input_mapping = [i for i in range(10)]
        if s == SIMPLE_LIST_STRATEGY:
            strat = strategy.SimpleListStrategy(
                self.unstructured_dataset, input_mapping, n_digits)
        return strat

    def generate_datapoint(self):
        imgs_lst = [None] * self.config[LENGTH]
        int_lst = [None] * self.config[LENGTH]
        for i in range(self.config[LENGTH]):
            samples = self.strat.sample()
            imgs, number_lst = zip(*samples)
            number = ''.join(str(n) for n in number_lst)
            imgs_lst[i] = imgs
            int_lst[i] = int(number)
        return (imgs_lst, int_lst)

    def generate_dataset(self):
        length = self.__len__()
        dataset = [None] * length
        for i in range(length):
            dataset[i] = self.generate_datapoint()

    def flatten(config, input):
        return [item for i in input for item in i]

    def unflatten(config, samples):
        result = [[] * config[LENGTH]]
        idx = 0
        for i in range(config[LENGTH]):
            number = [0] * config[N_DIGITS]
            for j in range(config[N_DIGITS]):
                number[j] = samples[idx]
                idx += 1
            result[i] = number
        return result

    def n_unflatten(config):
        return config[LENGTH] * config[N_DIGITS]


class StringDataset(StructuredDataset):

    def __init__(self, config, unstructured_dataset):
        self.config = config
        self.unstructured_dataset = unstructured_dataset
        # TODO: move this to the unstructured dataset
        self.input_mapping = ['0', '1', '2', '3', '4',
                              '5', '6', '7', '8', '9', '+', '-', '*', '/']
        self.strategy = self.get_strategy()
        self.dataset = self.generate_dataset()

    def __len__(self):
        return len(self.unstructured_dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch)

    def forward(net, x):
        return [net(item) for item in x]

    def get_strategy(self):
        s = self.config[STRATEGY]
        input_mapping = [i for i in range(len(self.unstructured_dataset))]

        if s == SINGLETON_STRATEGY:
            strat = strategy.SingletonStrategy(
                self.unstructured_dataset, input_mapping)
        return strat

    def generate_datapoint(self):
        return self.strategy.sample()

    def generate_dataset(self):
        length = self.__len__()
        dataset = [None] * length
        for i in range(length):
            dataset[i] = self.generate_datapoint()

    def flatten(config, input):
        return input

    def unflatten(config, samples):
        ud = get_unstructured_dataset_static(config)
        input_mapping = ud.input_mapping()
        string = ''
        for i in samples:
            string += input_mapping[0][i]
        return [string]

    def n_unflatten(config):
        return config[LENGTH]


def get_unstructured_dataset_static(config):
    if config[DATASET] == MNIST:
        return unstructured_dataset.MNISTDataset
    elif config[DATASET] == HWF_SYMBOL:
        return unstructured_dataset.HWFDataset
    elif config[DATASET] == MNIST_VIDEO:
        return unstructured_dataset.MNISTVideoDataset
    elif config[DATASET] == MNIST_GRID:
        return unstructured_dataset.MNISTGridDataset


def get_structured_dataset_static(config):
    if config[TYPE] == DIGIT_TYPE:
        return SingleIntDataset
    if config[TYPE] == INT_TYPE:
        return IntDataset
    elif config[TYPE] == SINGLE_INT_LIST_TYPE:
        return SingleIntListDataset
    elif config[TYPE] == INT_LIST_TYPE:
        return IntListDataset
    elif config[TYPE] == STRING_TYPE:
        return StringDataset
