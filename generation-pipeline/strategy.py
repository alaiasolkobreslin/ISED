
from constants import *
import random
import os
import numpy as np


class Strategy:

    def sample(self):
        """
        Returns a datapoint sampled according to the given strategy
        """
        pass


class SingleSampleStrategy(Strategy):
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


class Simple2DListStrategy(Strategy):

    def __init__(self, unstructured_dataset, input_mapping, n_rows, n_cols):
        self.unstructured_dataset = unstructured_dataset
        self.input_mapping = input_mapping
        self.n_rows = n_rows
        self.n_cols = n_cols

    def sample(self):
        samples = []
        for _ in range(self.n_rows):
            row_img = []
            row_value = []
            for _ in range(self.n_cols):
                sample = random.randrange(0, len(self.input_mapping))
                idx = self.unstructured_dataset.sample_with_y(sample)
                (img, value) = self.unstructured_dataset.get(idx)
                row_img.append(img)
                row_value.append(value)
            samples.append((row_img, row_value))
        return samples
