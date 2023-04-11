
from constants import *
from unstructured_dataset import *
from structured_dataset import *
import task_program


class TaskDataset:

    def __init__(self, config):
        self.config = config
        py_program = config[PY_PROGRAM]
        self.function = task_program.dispatcher[py_program]
        self.structured_dataset_train = {}
        self.structured_dataset_test = {}
        self.unstructured_datasets = {}
        inputs = self.config[INPUTS]
        for input in inputs:
            name = input[NAME]
            unstructured_dataset_train = TaskDataset.get_unstructured_dataset(
                input, train=True)
            unstructured_dataset_test = TaskDataset.get_unstructured_dataset(
                input, train=False)
            structured_dataset_train = self.get_structured_dataset(
                input, unstructured_dataset_train)
            structured_dataset_test = self.get_structured_dataset(
                input, unstructured_dataset_test)
            self.unstructured_datasets[name] = unstructured_dataset_train
            self.structured_dataset_train[name] = structured_dataset_train
            self.structured_dataset_test[name] = structured_dataset_test

    def __len__(self):
        # TODO: FIX
        return len(MNISTDataset(train=True).data)

    @staticmethod
    def collate(batch_dict):
        pass

    def get_unstructured_dataset(config, train):
        if config[DATASET] == MNIST:
            return MNISTDataset(train=train)
        elif config[DATASET] == HWF_SYMBOL:
            return HWFDataset(train=train)

    def get_structured_dataset(self, config, unstructured_dataset):
        if config[TYPE] == INT_TYPE:
            return SingleIntDataset(config, unstructured_dataset)
        elif config[TYPE] == INT_LIST_TYPE:
            return IntDataset(config, unstructured_dataset)
        elif config[TYPE] == STRING_TYPE:
            return StringDataset(config, unstructured_dataset)

    def generate_datapoint(self, train):
        prog = self.config[PY_PROGRAM]
        inputs = self.config[INPUTS]
        imgs = {}
        dispatch_args = {}
        for input in inputs:
            name = input[NAME]
            if train:
                (unstructured,
                 structured) = self.structured_dataset_train[name].generate_datapoint()
            else:
                (unstructured,
                 structured) = self.structured_dataset_test[name].generate_datapoint()
            imgs[name] = unstructured
            dispatch_args[name] = structured

        result = task_program.dispatch(prog, dispatch_args)
        return (imgs, self.unstructured_datasets, result)
