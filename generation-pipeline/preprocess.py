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
        length = len(input)
        indices = [i for i in range(length ** 2)]
        selected = np.random.choice(indices, 30, replace=False)
        bool_board = [[0 for _ in range(length)] for _ in range(length)]
        i = 0
        for i in range(length ** 2):
            row = i // length
            col = i % length
            if i in selected:
                curr_digit = input[row][1][col]
                input[row][1][col] = str(curr_digit)
                bool_board[row][col] = 1
            else:
                input[row][1][col] = '.'
        return (input, bool_board)
