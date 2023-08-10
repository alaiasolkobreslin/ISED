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
        # length = len(input)
        # (rows, cols) = bool_board.shape
        # # bool_board = [[0 for _ in range(length)] for _ in range(length)]
        # for i in range(length):
        #     for j in range(length):
        #         curr_digit = input[i][1][j]
        #         if curr_digit:
        #             input[i][1][j] = str(curr_digit)
        #             bool_board[i][j] = 1
        #         else:
        #             input[i][1][j] = '.'
        # return (input, bool_board)
        return (input, bool_board)


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


class PreprocessCoffee(Preprocess):

    def preprocess(self, input):
        return input
