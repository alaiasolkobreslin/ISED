import itertools
import math

import numpy as np


class Permute:

    def permutations(self, input):
        pass


class ListPermute(Permute):

    def permutations(self, lst):
        permutations = itertools.permutations(lst)
        return [list(p) for p in permutations]


class ListReverse(Permute):

    def permutations(self, lst):
        return [list(reversed(lst))]


class GridPermute(Permute):

    def permutations(self, grid):
        n = len(grid)
        permutations = [[]] * math.factorial(n*n)
        elements = [item for row in grid for item in row]
        for idx, permutation in enumerate(itertools.permutations(elements)):
            grid_permute = [[]] * n
            for i in range(n):
                row = [0] * n
                for j in range(n):
                    row[j] = permutation[i * n + j]
                grid_permute[i] = row
            permutations[idx] = grid_permute
        return permutations


class GridReflectHorizontal(Permute):

    def permutations(self, grid):
        return [reversed(grid)]


class GridReflectVertical(Permute):

    def permutations(self, grid):
        return [[reversed(row) for row in grid]]


class GridRotate90(Permute):

    def permutations(self, grid):
        return [np.rot90(grid).tolist()]


class GridRotate180(Permute):

    def permutations(self, grid):
        rot90 = np.rot90(grid)
        return [np.rot90(rot90).tolist()]


class GridRotate270(Permute):

    def permutations(self, grid):
        rot90 = np.rot90(grid)
        rot180 = np.rot90(rot90)
        return [np.rot90(rot180).tolist()]


list_permutations = [ListPermute, ListReverse]
grid_permutations = [GridPermute, GridReflectHorizontal,
                     GridReflectVertical, GridRotate90, GridRotate180, GridRotate270]
