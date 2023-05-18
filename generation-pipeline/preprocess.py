import numpy as np


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


class PreprocessSudokuBoard(Preprocess):

    def preprocess(self, input):
        # length = len(input)
        # TODO: make sudoku board
        return input
