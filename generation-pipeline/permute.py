import itertools
import math

import numpy as np

from constants import *


class Permute:

    def permutations(self, input):
        pass


class InputPermute(Permute):

    def __init__(self):
        self.does_permute = True

    def permutations(self, inputs):
        permutations = itertools.permutations(inputs)
        return [tuple(p) for p in permutations]


class ListPermute(Permute):

    def __init__(self):
        self.does_permute = True

    def permutations(self, lst):
        permutations = itertools.permutations(lst)
        return [list(p) for p in permutations]


class ListReverse(Permute):

    def __init__(self):
        self.does_permute = True

    def permutations(self, lst):
        return [list(reversed(lst))]


class ListListPermute(Permute):

    def __init__(self):
        self.does_permute = True

    def permutations(self, input):
        n = len(input)
        m = len(input[0])
        permutations = [[]] * math.factorial(n*m)
        elements = [item for row in input for item in row]
        for idx, permutation in enumerate(itertools.permutations(elements)):
            list_list_permute = [[]] * n
            for i in range(n):
                row = [0] * m
                for j in range(n):
                    row[j] = permutation[i * n + j]
                list_list_permute[i] = row
            permutations[idx] = list_list_permute
        return permutations


class ListListReflectHorizontal(Permute):

    def __init__(self):
        self.does_permute = True

    def permutations(self, grid):
        return [reversed(grid)]


class ListListReflectVertical(Permute):

    def __init__(self):
        self.does_permute = True

    def permutations(self, grid):
        return [[reversed(row) for row in grid]]


class GridRotate90(Permute):

    def __init__(self):
        self.does_permute = True

    def permutations(self, grid):
        return [np.rot90(grid).tolist()]


class GridRotate180(Permute):

    def __init__(self):
        self.does_permute = True

    def permutations(self, grid):
        rot90 = np.rot90(grid)
        return [np.rot90(rot90).tolist()]


class GridRotate270(Permute):

    def __init__(self):
        self.does_permute = True

    def permutations(self, grid):
        rot90 = np.rot90(grid)
        rot180 = np.rot90(rot90)
        return [np.rot90(rot180).tolist()]


class AllPermutations:

    def __init__(self, inputs_config):
        self.input_permutations = self.get_input_permutations(inputs_config)
        self.perm_dict = self.get_perm_dict(inputs_config)
        self.inputs_config = inputs_config

    def get_input_permutations(self, inputs_config):
        return [InputPermute]

    def get_perm_dict(self, inputs_config):
        perm_dict = {}
        for input in inputs_config:
            input_type = input[TYPE]
            if input_type in [DIGIT_TYPE, CHAR_TYPE]:
                permutations = []
            elif input_type in [INT_TYPE, SINGLE_INT_LIST_TYPE, INT_LIST_TYPE, STRING_TYPE]:
                permutations = [ListPermute, ListReverse]
            elif input_type in [SINGLE_INT_LIST_LIST_TYPE]:
                permutations = [
                    ListListPermute, ListListReflectHorizontal, ListListReflectVertical]
            elif input_type in [SINGLE_INT_GRID_TYPE]:
                permutations = [ListListPermute, ListListReflectHorizontal,
                                ListListReflectVertical, GridRotate90, GridRotate180, GridRotate270]
            perm_dict[input[NAME]] = permutations
        return perm_dict

    def permute_all_inputs(self, inputs):
        permutations = []
        # First permute the inputs as a whole
        for input_permutation in self.input_permutations:
            permutations += input_permutation().permutations(inputs)
        # Next, keep inputs in order while permuting them individually
        for i, input in enumerate(self.inputs_config):
            for perm in self.perm_dict[input[NAME]]:
                for p in perm().permutations(inputs[i]):
                    permuted_inputs = [j for j in inputs]
                    permuted_inputs[i] = p
                    permutations.append(tuple(permuted_inputs))
        res = []
        [res.append(x) for x in permutations if x not in res]
        return res