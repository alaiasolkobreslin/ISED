from constants import *

import strategy


class StructuredDataset:

    def __len__(self):
        pass

    def generate_datapoint(self):
        pass

    def get_strategy(self):
        pass

    def generate_dataset(self):
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
        self.strategy = self.get_strategy()
        self.dataset = self.generate_dataset()

    def __len__(self):
        return int(len(self.unstructured_dataset) / self.config[N_DIGITS])

    def __getitem__(self, index):
        return self.dataset[index]

    def get_strategy(self):
        n_digits = self.config[N_DIGITS]
        s = self.config[STRATEGY]
        input_mapping = [i for i in range(10)]
        if s == SIMPLE_LIST_STRATEGY:
            strat = strategy.SimpleListStrategy(
                self.unstructured_dataset, input_mapping, n_digits)
        return strat

    def generate_datapoint(self):
        samples = self.strat.sample()
        imgs, number_lst = zip(*samples)
        number = ''.join(str(n) for n in number_lst)
        return (imgs, int(number))

    def generate_dataset(self):
        length = self.__len__()
        dataset = [None] * length
        for i in range(length):
            dataset[i] = self.generate_datapoint()

    def flatten(self, input):
        return input

    def unflatten(self, samples):
        number = ''
        for i in samples:
            number += str(i)
        return [number]


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

    def get_strategy(self):
        length = self.config[LENGTH]
        s = self.config[STRATEGY]
        input_mapping = [i for i in range(10)]
        if s == SIMPLE_LIST_STRATEGY:
            strat = strategy.SimpleListStrategy(
                self.unstructured_dataset, input_mapping, length)
        return strat

    def generate_datapoint(self):
        samples = self.strat.sample()
        return zip(*samples)

    def generate_dataset(self):
        length = self.__len__()
        dataset = [None] * length
        for i in range(length):
            dataset[i] = self.generate_datapoint()

    def flatten(self, input):
        return input

    def unflatten(self, samples):
        return samples


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

    def flatten(self, input):
        return [item for i in input for item in i]

    def unflatten(self, samples):
        result = [[] * self.config[LENGTH]]
        idx = 0
        for i in range(self.config[LENGTH]):
            number = [0] * self.config[N_DIGITS]
            for j in range(self.config[N_DIGITS]):
                number[j] = samples[idx]
                idx += 1
            result[i] = number
        return result


class StringDataset(StructuredDataset):

    def __init__(self, config, unstructured_dataset):
        self.config = config
        self.unstructured_dataset = unstructured_dataset
        self.input_mapping = ['0', '1', '2', '3', '4',
                              '5', '6', '7', '8', '9', '+', '-', '*', '/']
        self.strategy = self.get_strategy()
        self.dataset = self.generate_dataset()

    def __len__(self):
        return len(self.unstructured_dataset)

    def __getitem__(self, index):
        return self.dataset[index]

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

    def flatten(self, input):
        return input

    def unflatten(self, samples):
        string = ''
        for i in samples:
            string += self.input_mapping[i]
        return [string]
