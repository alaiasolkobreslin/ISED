from constants import *
from unstructured_dataset import *
from structured_dataset import *
import task_program


class TaskDataset:

    def __init__(self, config, train):
        self.config = config
        py_program = config[PY_PROGRAM]
        self.function = task_program.dispatcher[py_program]
        self.structured_datasets = {}
        inputs = self.config[INPUTS]
        for input in inputs:
            name = input[NAME]
            unstructured_dataset = TaskDataset.get_unstructured_dataset(
                input, train=train)
            structured_dataset = TaskDataset.get_structured_dataset(
                input, unstructured_dataset)
            self.structured_datasets[name] = structured_dataset
        self.dataset = self.generate_task_dataset()

    def __len__(self):
        return min(sd.__len__() for sd in self.structured_datasets.values())

    def __getitem__(self, index):
        return self.dataset[index]

    def get_unstructured_dataset(config, train):
        dataset = get_unstructured_dataset_static(config)
        return dataset(train=train)

    def get_structured_dataset(config, unstructured_dataset):
        dataset = get_structured_dataset_static(config)
        return dataset(config, unstructured_dataset)

    def generate_datapoint(self):
        prog = self.config[PY_PROGRAM]
        inputs = self.config[INPUTS]
        imgs = {}
        dispatch_args = {}
        for input in inputs:
            name = input[NAME]
            (unstructured,
             structured) = self.structured_datasets[name].generate_datapoint()
            imgs[name] = unstructured
            dispatch_args[name] = structured

        result = task_program.dispatch(prog, dispatch_args)
        return (imgs, result)

    def generate_task_dataset(self):
        length = self.__len__()
        dataset = [None] * length
        for i in range(length):
            dataset[i] = self.generate_datapoint()
        return dataset

    def __getitem__(self, index):
        return self.dataset[index]
