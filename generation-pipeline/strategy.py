
from constants import *
import random


class Strategy:

    def sample(self):
        """
        Returns a datapoint sampled according to the given strategy
        """
        pass


class SingletonStrategy(Strategy):
    def __init__(self, unstructured_dataset, input_mapping):
        self.unstructured_dataset = unstructured_dataset
        self.input_mapping = input_mapping

    def sample(self):
        sample = random.randrange(0, len(self.input_mapping))
        idx = self.unstructured_dataset.sample_with_y(sample)
        return self.unstructured_dataset.get(idx)


class SimpleListStrategy(Strategy):

    def __init__(self, unstructured_dataset, input_mapping, n_samples):
        self.unstructured_dataset = unstructured_dataset
        self.input_mapping = input_mapping
        self.n_samples = n_samples

    def sample(self):
        samples = [None] * self.n_samples
        for i in range(self.n_samples):
            sample = random.randrange(0, len(self.input_mapping))
            idx = self.unstructured_dataset.sample_with_y(sample)
            samples[i] = self.unstructured_dataset.get(idx)
        return samples
