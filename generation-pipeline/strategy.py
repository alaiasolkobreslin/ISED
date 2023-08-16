
from constants import *
import random
import os
import numpy as np


class Strategy:

    def sample(self):
        """
        Returns a datapoint sampled according to the given strategy
        """
        pass


class SingleSampleStrategy(Strategy):
    def __init__(self, unstructured_dataset, input_mapping):
        self.unstructured_dataset = unstructured_dataset
        self.input_mapping = input_mapping

    def sample(self):
        sample = random.randrange(0, len(self.input_mapping))
        idx = self.unstructured_dataset.sample_with_y(sample)
        return self.unstructured_dataset.get(idx)


class SimpleListStrategy(Strategy):

    def __init__(self, unstructured_dataset, input_mapping, n_samples):
        self.unstructured_dataset = unstructured_dataset
        self.input_mapping = input_mapping
        self.n_samples = n_samples

    def sample(self):
        samples = [None] * self.n_samples
        for i in range(self.n_samples):
            sample = random.randrange(0, len(self.input_mapping))
            idx = self.unstructured_dataset.sample_with_y(sample)
            samples[i] = self.unstructured_dataset.get(idx)
        return samples


class Simple2DListStrategy(Strategy):

    def __init__(self, unstructured_dataset, input_mapping, n_rows, n_cols):
        self.unstructured_dataset = unstructured_dataset
        self.input_mapping = input_mapping
        self.n_rows = n_rows
        self.n_cols = n_cols

    def sample(self):
        samples = []
        for _ in range(self.n_rows):
            row_img = []
            row_value = []
            for _ in range(self.n_cols):
                sample = random.randrange(0, len(self.input_mapping))
                idx = self.unstructured_dataset.sample_with_y(sample)
                (img, value) = self.unstructured_dataset.get(idx)
                row_img.append(img)
                row_value.append(value)
            samples.append((row_img, row_value))
        return samples


class SudokuProblemStrategy(Strategy):

    def __init__(self, unstructured_dataset, n_rows, n_cols):
        self.unstructured_dataset = unstructured_dataset
        self.n_rows = n_rows
        self.n_cols = n_cols
        if self.n_rows == 4:
            self.input_mapping = [i for i in range(1, 5)]
            self.file = 'data/Sudoku/4x4_sudoku_unique_puzzles.csv'
        elif self.n_rows == 9:
            self.input_mapping = [i for i in range(1, 10)]
            self.file = 'data/Sudoku/sudoku.csv'
        self.quizzes, self.solutions = self.quizzes_and_solutions()
        self.n_problems = 1000000

    def quizzes_and_solutions(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        sudoku_path = os.path.join(dir_path, self.file)

        total_squares = self.n_rows * self.n_cols
        quizzes = np.zeros((1000000, total_squares), np.int32)
        solutions = np.zeros((1000000, total_squares), np.int32)
        for i, line in enumerate(open(sudoku_path, 'r').read().splitlines()[1:]):
            quiz, solution = line.split(",")
            for j, q_s in enumerate(zip(quiz, solution)):
                q, s = q_s
                quizzes[i, j] = q
                solutions[i, j] = s
        quizzes = quizzes.reshape((-1, self.n_rows, self.n_cols))
        solutions = solutions.reshape((-1, self.n_rows, self.n_cols))
        return quizzes, solutions

    def sample(self):
        problem_sample = random.randrange(0, self.n_problems)
        sampled_problem = self.quizzes[problem_sample]
        bool_board = np.copy(sampled_problem)
        bool_board[bool_board != 0] = 1
        samples = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                sample = sampled_problem[i][j]
                if sample != 0:
                    idx = self.unstructured_dataset.sample_with_y(sample)
                    (img, value) = self.unstructured_dataset.get(idx)
                    samples.append((img, value))
        return (samples, len(samples), bool_board)


class SudokuRandomStrategy(Strategy):

    def __init__(self, unstructured_dataset, n_rows, n_cols):
        self.problem_strategy = SudokuProblemStrategy(
            unstructured_dataset, n_rows, n_cols)
        self.unstructured_dataset = unstructured_dataset
        self.n_rows = n_rows
        self.n_cols = n_cols

    def sample(self):
        if random.randint(0, 1):
            # half of sudoku board should be valid
            return self.problem_strategy.sample()
        else:
            # the other half should be randomly generated (likely invalid)
            flag = True
            while flag:
                problem = []
                for _ in range(self.n_rows):
                    problem_row = []
                    for _ in range(self.n_cols):
                        if random.randint(1, 81) < 30:
                            sample = random.randrange(1, 1 + self.n_rows)
                            flag = False
                        else:
                            sample = 0
                        problem_row.append(sample)
                    problem.append(problem_row)
            problem = np.array(problem)
            bool_board = np.copy(problem)
            bool_board[bool_board != 0] = 1

            samples = []
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    sample = problem[i][j]
                    if sample != 0:
                        idx = self.unstructured_dataset.sample_with_y(sample)
                        (img, value) = self.unstructured_dataset.get(idx)
                        samples.append((img, value))
            return (samples, len(samples), bool_board)
