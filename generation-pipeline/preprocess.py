import numpy as np
import random


class Preprocess:

    def preprocess(input):
        pass


class PreprocessIdentity(Preprocess):

    def preprocess(self, input):
        return input


class PreprocessSort(Preprocess):

    def preprocess(self, input):
        ys = [y for (_, y) in input]
        idxs = np.argsort(ys)
        return [input[i] for i in idxs]
