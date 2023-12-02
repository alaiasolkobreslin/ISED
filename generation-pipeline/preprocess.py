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


class PreprocessSudokuBoard(Preprocess):

    def preprocess(self, input, bool_board):
        length = bool_board.shape[0]
        board = []
        idx = 0
        for i in range(length):
            row = []
            for j in range(length):
                if bool_board[i][j]:
                    row.append(str(input[idx][1]))
                    idx += 1
                else:
                    row.append('.')
            board.append(row)
        return (input, board)


class PreprocessPalindrome(Preprocess):

    def ceildiv(self, a, b):
        return -(a // -b)

    def preprocess(self, input):
        if random.randint(0, 1):
            return input
        else:
            half_to_append = input[:(len(input) // 2)]
            half_to_return = input[:self.ceildiv(len(input), 2)]
            return half_to_return + list(reversed(half_to_append))
