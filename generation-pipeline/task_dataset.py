from constants import *
from unstructured_dataset import *
from structured_dataset import *
import task_program


class TaskDataset:

    def __init__(self, config, train):
        self.config = config
        py_program = config[PY_PROGRAM]
        self.dataset_size = config[DATASET_SIZE_TRAIN] if train else config[DATASET_SIZE_TEST]
        self.function = task_program.dispatcher[py_program]
        self.structured_datasets = {}
        inputs = self.config[INPUTS]
        self.no_dataset_generation = config[NO_DATASET_GENERATION] \
            if NO_DATASET_GENERATION in config else False
        if self.no_dataset_generation:
            self.unstructured_dataset = TaskDataset.get_unstructured_dataset(
                inputs[0], train=train)
        for input in inputs:
            name = input[NAME]
            unstructured_dataset = TaskDataset.get_unstructured_dataset(
                input, train=train)
            structured_dataset = TaskDataset.get_structured_dataset(
                input, self.dataset_size, unstructured_dataset)
            self.structured_datasets[name] = structured_dataset
        self.dataset = self.generate_task_dataset()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return self.dataset[index]

    def get_unstructured_dataset(config, train):
        """
        Returns the unstructured dataset for `config` with training mode set to
        `train`
        """
        dataset = get_unstructured_dataset_static(config)
        return dataset(train=train)

    def get_structured_dataset(config, dataset_size, unstructured_dataset):
        """
        Returns the structured dataset for `config` and `unstructured_dataset`
        """
        dataset = get_structured_dataset_static(config)
        return dataset(config, dataset_size, unstructured_dataset)

    def generate_datapoint(self):
        """
        Generates datapoints for each input to the task program and executes the
        given program on these inputs to compute the ground truth result.

        Returns a datapoint consisting of the unstructured images, the input configurations,
        and the ground truth result.
        """
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

        if self.structured_datasets[name].call_black_box_for_gt:
            result = task_program.dispatch(prog, dispatch_args)
        else:
            result = structured
        return (imgs, self.config[INPUTS], result)

    def generate_task_dataset(self):
        """
        Returns a complete dataset for this task program by repeatedly making
        calls to `generate_datapoint` and storing the resulting datapoints in
        `dataset` before returning it
        """
        if self.no_dataset_generation:
            if self.unstructured_dataset.name == HWF_SYMBOL:
                original_dataset = self.unstructured_dataset.get_full_dataset()
                length = len(original_dataset)
                dataset = [None] * length
                name = self.config[INPUTS][0][NAME]
                for i in range(length):
                    (imgs, expr, result) = original_dataset[i]
                    dataset[i] = ({name: imgs}, self.config[INPUTS], result)
                return dataset
            elif self.unstructured_dataset == COFFEE_LEAF_MINER or self.unstructured_dataset == COFFEE_LEAF_RUST:
                original_dataset = self.unstructured_dataset.get_full_dataset()
                length = len(original_dataset)
                dataset = [None] * length
                name = self.config[INPUTS][0][NAME]
                for i in range(length):
                    (imgs, result) = original_dataset[i]
                    dataset[i] = ({name: imgs}, self.config[INPUTS], result)
                return dataset
            else:
                raise Exception(f"Unsupported dataset {self.unstructured_dataset.name}")
        else:
            length = self.__len__()
            dataset = [None] * length
            for i in range(length):
                dataset[i] = self.generate_datapoint()
            return dataset
