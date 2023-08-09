from typing import *

import torch

from constants import *
import util


class OutputMapping:
    def __init__(self): pass

    def vectorize(self, results: List, result_probs: torch.Tensor):
        """
        An output mapping should implement this function to vectorize the results and result probabilities
        """
        pass

    def get_normalized_labels(self, y_pred, target, output_mapping):
        """
        Return the normalized labels to be used in the loss function
        """
        pass

    def eval_result_eq(self, a, b, threshold=0.01):
        """
        Returns True if two results are equal and False otherwise
        """
        if type(a) is float or type(b) is float:
            result = abs(a - b) < threshold
        elif type(a) is tuple and type(b) is tuple:
            result = True
            for i in range(len(a)):
                result = result and self.eval_result_eq(a[i], b[i], threshold)
        else:
            result = a == b
        return result


class DiscreteOutputMapping(OutputMapping):
    def __init__(self, elements: List[Any]):
        self.elements = elements
        self.element_indices = {e: i for (i, e) in enumerate(elements)}

    def vectorize(self, results: List, result_probs: torch.Tensor) -> torch.Tensor:
        batch_size, sample_count = result_probs.shape
        result_tensor = torch.zeros((batch_size, len(self.elements)))
        for i in range(batch_size):
            for j in range(sample_count):
                if results[i][j] != RESERVED_FAILURE:
                    result_tensor[i, self.element_indices[results[i]
                                                          [j]]] += result_probs[i, j]
        return (self.element_indices, torch.nn.functional.normalize(result_tensor, dim=1))

    def get_normalized_labels(self, y_pred, target, output_mapping):
        batch_size, _ = y_pred.shape
        y = torch.tensor([1.0 if self.eval_result_eq(
            util.get_hashable_elem(l), m) else 0.0 for l in target for m in output_mapping]).view(batch_size, -1)
        return y


class UnknownDiscreteOutputMapping(OutputMapping):
    def __init__(self, fallback):
        self.fallback = fallback

    def vectorize(self, results: List, result_probs: torch.Tensor) -> torch.Tensor:
        batch_size, sample_count = result_probs.shape

        # Get the unique elements
        elements = list(
            set([(util.get_hashable_elem(elem)) for batch in results for elem in batch if elem != RESERVED_FAILURE]))
        element_indices = {e: i for (i, e) in enumerate(elements)}

        # If there is no element being derived...
        if len(elements) == 0:
            # We return a single fallback value, while the probability of result being fallback are all 0
            return ([self.fallback], torch.tensor([[0.0]] * batch_size, requires_grad=True))

        # Vectorize the results
        result_tensor = torch.zeros((batch_size, len(elements)))
        for i in range(batch_size):
            for j in range(sample_count):
                if results[i][j] != RESERVED_FAILURE:
                    idx = util.get_hashable_elem(results[i][j])
                    result_tensor[i, element_indices[idx]
                                  ] += result_probs[i, j]
        result_tensor = torch.nn.functional.normalize(result_tensor, dim=1)

        # Return the elements mapping and also the result probability tensor
        return (elements, result_tensor)

    def get_normalized_labels(self, y_pred, target, output_mapping):
        batch_size, _ = y_pred.shape
        y = torch.tensor([1.0 if self.eval_result_eq(
            util.get_hashable_elem(l), m) else 0.0 for l in target for m in output_mapping]).view(batch_size, -1)
        return y


class ListOutputMapping(OutputMapping):
    def __init__(self, fallback):
        self.fallback = fallback

    def vectorize(self, results: List, result_probs: torch.Tensor) -> torch.Tensor:
        batch_size, sample_count = result_probs.shape

        result_tensor = torch.zeros((batch_size, 5, 10))
        for i, result in enumerate(results):
            for j, r in enumerate(result):
                r = str(r).rjust(5, "0")
                result_prob = result_probs[i][j]
                for idx in range(5):
                    elt = int(r[idx])
                    result_tensor[i][idx][elt] += result_prob

        result_tensor = torch.nn.functional.normalize(result_tensor, dim=2)

        elements = list(
            set([(util.get_hashable_elem(elem)) for batch in results for elem in batch if elem != RESERVED_FAILURE]))
        element_indices = {e: i for (i, e) in enumerate(elements)}

        # If there is no element being derived...
        if len(elements) == 0:
            # We return a single fallback value, while the probability of result being fallback are all 0
            return ([self.fallback], torch.tensor([[0.0]] * batch_size, requires_grad=True))

        # Vectorize the results
        result_tensor_old = torch.zeros((batch_size, len(elements)))
        for i in range(batch_size):
            for j in range(sample_count):
                if results[i][j] != RESERVED_FAILURE:
                    idx = util.get_hashable_elem(results[i][j])
                    result_tensor_old[i, element_indices[idx]
                                      ] += result_probs[i, j]
        result_tensor_old = torch.nn.functional.normalize(
            result_tensor_old, dim=1)

        # Return the elements mapping and also the result probability tensor
        return (elements, result_tensor, result_tensor_old)

    def get_normalized_labels(self, y_pred, target, output_mapping):
        batch_size = y_pred.shape[0]
        y = torch.zeros((batch_size, 5, 10))
        for i, l in enumerate(target):
            l = str(l).rjust(5, "0")
            for idx in range(5):
                elt = int(l[idx])
                y[i][idx][elt] = 1.0

        old_norm_label = torch.tensor([1.0 if self.eval_result_eq(
            util.get_hashable_elem(l), m) else 0.0 for l in target for m in output_mapping]).view(batch_size, -1)

        return y, old_norm_label


class SudokuOutputMapping(OutputMapping):
    def __init__(self, fallback):
        self.fallback = fallback

    def vectorize(self, results: List, result_probs: torch.Tensor) -> torch.Tensor:
        batch_size, sample_count = result_probs.shape

        result_tensor = torch.zeros((batch_size, 4, 4, 4))
        for i, result in enumerate(results):
            for j, r in enumerate(result):
                result_prob = result_probs[i][j]
                for row in range(4):
                    for col in range(4):
                        elt = r[row][col]
                        if elt in ['1', '2', '3', '4']:
                            idx = int(elt) - 1
                            result_tensor[i][row][col][idx] += result_prob

        result_tensor = torch.nn.functional.normalize(result_tensor, dim=2)

        elements = list(
            set([(util.get_hashable_elem(elem)) for batch in results for elem in batch if elem != RESERVED_FAILURE]))
        element_indices = {e: i for (i, e) in enumerate(elements)}

        # If there is no element being derived...
        if len(elements) == 0:
            # We return a single fallback value, while the probability of result being fallback are all 0
            return ([self.fallback], torch.tensor([[0.0]] * batch_size, requires_grad=True))

        # Vectorize the results
        result_tensor_old = torch.zeros((batch_size, len(elements)))
        for i in range(batch_size):
            for j in range(sample_count):
                if results[i][j] != RESERVED_FAILURE:
                    idx = util.get_hashable_elem(results[i][j])
                    result_tensor_old[i, element_indices[idx]
                                      ] += result_probs[i, j]
        result_tensor_old = torch.nn.functional.normalize(
            result_tensor_old, dim=1)

        # Return the elements mapping and also the result probability tensor
        return (elements, result_tensor, result_tensor_old)

    def get_normalized_labels(self, y_pred, target, output_mapping):
        batch_size = y_pred.shape[0]
        y = torch.zeros((batch_size, 4, 4, 4))
        for i, l in enumerate(target):
            for row in range(4):
                for col in range(4):
                    elt = l[row][col]
                    if elt in ['1', '2', '3', '4']:
                        idx = int(elt) - 1
                        y[i][row][col][idx] = 1.0

        old_norm_label = torch.tensor([1.0 if self.eval_result_eq(
            util.get_hashable_elem(l), m) else 0.0 for l in target for m in output_mapping]).view(batch_size, -1)

        return y, old_norm_label


def get_output_mapping(output_config):
    om = output_config[OUTPUT_MAPPING]
    if om == UNKNOWN:
        # return UnknownDiscreteOutputMapping(
        #     fallback=0)
        return ListOutputMapping(fallback=0)
    elif om == SUDOKU_OUTPUT_MAPPING:
        return SudokuOutputMapping(fallback=0)
    else:
        raise Exception(f"Unknown output mapping {om}")
