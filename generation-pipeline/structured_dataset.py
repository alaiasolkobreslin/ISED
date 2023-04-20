from constants import *

import strategy


class StructuredDataset:

    def generate_datapoint(self):
        pass

    def flatten(self, input):
        pass

    def unflatten(self, samples):
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

    def flatten(self, input):
        return [input]

    def unflatten(self, samples):
        return samples


class IntDataset(StructuredDataset):

    def __init__(self, config, unstructured_dataset):
        self.config = config
        self.unstructured_dataset = unstructured_dataset

    def generate_datapoint(self):
        n_digits = self.config[N_DIGITS]
        s = self.config[STRATEGY]
        input_mapping = [i for i in range(10)]

        if s == SIMPLE_LIST_STRATEGY:
            strat = strategy.SimpleListStrategy(
                self.unstructured_dataset, input_mapping, n_digits)

        samples = strat.sample()
        number = ""
        imgs = [None] * n_digits
        for (i, (img, digit)) in enumerate(samples):
            number += str(digit)
            imgs[i] = img
        return (imgs, int(number))

    def flatten(self, input):
        return input

    def unflatten(self, samples):
        number = ''
        for i in samples:
            number += str(i)
        return [number]


class IntListDataset(StructuredDataset):

    def __init__(self, config, unstructured_dataset):
        self.config = config
        self.unstructured_dataset = unstructured_dataset

    def generate_datapoint(self):
        # imgs_lst = [None] * self.config[LENGTH]
        # int_lst = [None] * self.config[LENGTH]

        # n_digits = self.config[N_DIGITS]
        # s = self.config[STRATEGY]
        # input_mapping = [i for i in range(0, 10)]

        # TODO
        pass


class StringDataset(StructuredDataset):

    def __init__(self, config, unstructured_dataset):
        self.config = config
        self.unstructured_dataset = unstructured_dataset
        self.input_mapping = ['0', '1', '2', '3', '4',
                              '5', '6', '7', '8', '9', '+', '-', '*', '/']

    def generate_datapoint(self):
        s = self.config[STRATEGY]
        input_mapping = [i for i in range(len(self.unstructured_dataset))]

        if s == SINGLETON_STRATEGY:
            strat = strategy.SingletonStrategy(
                self.unstructured_dataset, input_mapping)

        return strat.sample()

    def flatten(self, input):
        return input

    def unflatten(self, samples):
        string = ''
        for i in samples:
            string += self.input_mapping[i]
        return [string]
