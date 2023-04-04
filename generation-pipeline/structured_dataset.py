from constants import *

import strategy

class StructuredDataset:
    
    def generate_datapoint():
        pass

class SingleIntDataset(StructuredDataset):

    def __init__(self, config, unstructured_dataset):
        self.config = config
        self.unstructured_dataset = unstructured_dataset

    def generate_datapoint(self):
        pass
    

class IntDataset(StructuredDataset):
    
    def __init__(self, config, unstructured_dataset):
        self.config = config
        self.unstructured_dataset = unstructured_dataset
    
    def generate_datapoint(self) :
        n_digits = self.config[N_DIGITS]
        s = self.config[STRATEGY]
        input_mapping = [i for i in range(0, 10)]

        strat = strategy.get_strategy(s, self.unstructured_dataset, input_mapping, n_digits)
        samples = strat.sample()
        number = ""
        imgs = [None] * n_digits
        for (i, (img, digit)) in enumerate(samples):
            number += str(digit)
            imgs[i] = img
        return (imgs, int(number))

class IntListDataset(StructuredDataset):
    
    def __init__(self, config, unstructured_dataset):
        self.config = config
        self.unstructured_dataset = unstructured_dataset
    
    def generate_datapoint(self) :
        # imgs_lst = [None] * self.config[LENGTH]
        # int_lst = [None] * self.config[LENGTH]

        # n_digits = self.config[N_DIGITS]
        # s = self.config[STRATEGY]
        # input_mapping = [i for i in range(0, 10)]

        # TODO
        pass