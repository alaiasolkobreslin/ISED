
from constants import *
from unstructured_dataset import *
from structured_dataset import *
import task_program

class TaskDataset:
    
    def __init__(self, config):
        self.config = config
        py_program = config[PY_PROGRAM]
        self.function = task_program.dispatcher[py_program]

    def get_unstructured_dataset(self, config):
        if config[DATASET] == MNIST:
            return MNISTDataset()
        
    def get_structured_dataset(self, config, unstructured_dataset):
        if config[TYPE] == INT_TYPE:
            return SingleIntDataset(config, unstructured_dataset)
        if config[TYPE] == INT_LIST_TYPE:
            return IntDataset(config, unstructured_dataset)

    def generate_datapoint(self):
        prog = self.config[PY_PROGRAM]
        inputs = self.config[INPUTS]
        imgs = {}
        dispatch_args = {}
        for input in inputs:
            name = input[NAME]
            unstructured_dataset = self.get_unstructured_dataset(input)
            structured_dataset = self.get_structured_dataset(input, unstructured_dataset)
            (unstructured, structured) = structured_dataset.generate_datapoint()
            imgs[name] = unstructured
            dispatch_args[name] = structured

        result = task_program.dispatch(prog, dispatch_args)
        return (imgs, result)
