
from constants import *
from unstructured_dataset import *
import task_program

class TaskDataset:
    
    def __init__(self, config):
        self.config = config
        py_program = config[PY_PROGRAM]
        self.function = task_program.dispatcher[py_program]

    def get_unstructured_dataset(dataset):
        if dataset == MNIST:
            return MNISTDataset()

    def generate_datapoint(self):
        inputs = self.config[INPUTS]
        for input in inputs:
            name = input[NAME]
            unstructured_dataset = self.get_unstructured_dataset(input[DATASET])
        pass

