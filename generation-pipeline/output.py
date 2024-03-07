from typing import *

import torch

from constants import *
import util
from frozendict import frozendict

class OutputMapping:
    def __init__(self): pass

    def dim(self): raise Exception(f"Non-implemented dim for {self.__class__}")

    def get_elements(self): pass

    def vectorize(self, elements, element_indices: dict, results: List, result_probs: torch.Tensor):
        """
        An output mapping should implement this function to vectorize the results and result probabilities
        """
        batch_size, sample_count = result_probs.shape
        result_tensor = torch.zeros((batch_size, len(elements)))
        for i in range(batch_size):
            for j in range(sample_count):
                if results[i][j] != RESERVED_FAILURE:
                    idx = util.get_hashable_elem(results[i][j])
                    result_tensor[i, element_indices[idx]
                                  ] += result_probs[i, j]
        return (element_indices, torch.nn.functional.normalize(result_tensor, dim=1))

    def get_normalized_labels(self, y_pred, target, output_mapping):
        """
        Return the normalized labels to be used in the loss function
        """
        batch_size = y_pred.shape[0]
        y = torch.tensor([1.0 if self.eval_result_eq(
            util.get_hashable_elem(l), m) else 0.0 for l in target for m in output_mapping]).view(batch_size, -1)
        return y

    def eval_result_eq(self, a, b, threshold=0.01):
        """
        Returns True if two results are equal and False otherwise
        """
        if type(a) is float or type(b) is float:
            result = abs(a - b) < threshold
        elif type(a) is tuple and type(b) is tuple:
            if len(a) != len(b):
                return False
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

    def dim(self):
        return len(self.elements)

    def vectorize(self, results: List, result_probs: torch.Tensor) -> torch.Tensor:
        batch_size, sample_count = result_probs.shape
        result_tensor = torch.zeros((batch_size, len(self.elements)))
        for i in range(batch_size):
            for j in range(sample_count):
                if results[i][j] != RESERVED_FAILURE:
                    result_tensor[i, self.element_indices[results[i]
                                                          [j]]] += result_probs[i, j]
        y_pred = torch.nn.functional.normalize(result_tensor, dim=1)
        return (self.element_indices, y_pred, y_pred)

    def vectorize_label(self, labels):
        return torch.stack([
            torch.tensor([1.0 if self.eval_result_eq(e, label)
                         else 0.0 for e in self.elements])
            for label in labels])

    def get_normalized_labels(self, y_pred, target, output_mapping):
        y = super().get_normalized_labels(y_pred, target, output_mapping)
        return (y, y)


class UnknownDiscreteOutputMapping(OutputMapping):
    def __init__(self, fallback):
        self.fallback = fallback

    def vectorize(self, results: List, result_probs: torch.Tensor) -> torch.Tensor:
        # Get the unique elements
        elements = list(
            set([(util.get_hashable_elem(elem)) for batch in results for elem in batch if elem != RESERVED_FAILURE]))
        element_indices = {e: i for (i, e) in enumerate(elements)}

        om, y_pred = super().vectorize(elements, element_indices, results, result_probs)
        return (om, y_pred, y_pred)

    def get_normalized_labels(self, y_pred, target, output_mapping):
        batch_size, _ = y_pred.shape
        y = torch.tensor([1.0 if self.eval_result_eq(
            util.get_hashable_elem(l), m) else 0.0 for l in target for m in output_mapping]).view(batch_size, -1)
        return (y, y)

class RangeOutputMapping(OutputMapping):
    def __init__(self, range_low, range_high):
        self.range_low = range_low
        self.range_high = range_high

    def dim(self):
        return self.range_high - self.range_low

    def vectorize(self, results: List, result_probs: torch.Tensor) -> torch.Tensor:
        # Get the unique elements
        elements = list(
            set([(util.get_hashable_elem(elem)) for batch in results for elem in batch if elem != RESERVED_FAILURE]))
        element_indices = {e: i for (i, e) in enumerate(elements)}

        om, y_pred = super().vectorize(elements, element_indices, results, result_probs)
        return (om, y_pred, y_pred)

    def vectorize_label(self, labels):
        return torch.stack([
            torch.tensor([1.0 if self.eval_result_eq(e, label)
                         else 0.0 for e in range(self.range_low, self.range_high)])
            for label in labels])

    def get_normalized_labels(self, y_pred, target, output_mapping):
        batch_size, _ = y_pred.shape
        y = torch.tensor([1.0 if self.eval_result_eq(
            util.get_hashable_elem(l), m) else 0.0 for l in target for m in output_mapping]).view(batch_size, -1)
        return (y, y)


class IntOutputMapping(OutputMapping):
    def __init__(self, length, n_classes, fallback):
        self.length = length
        self.n_classes = n_classes
        self.fallback = fallback

    def vectorize(self, results: List, result_probs: torch.Tensor) -> torch.Tensor:
        batch_size, _ = result_probs.shape

        elements = list(
            set([(util.get_hashable_elem(elem)) for batch in results for elem in batch if elem != RESERVED_FAILURE]))
        element_indices = {e: i for (i, e) in enumerate(elements)}
        if len(elements) == 0:
            return ([self.fallback], torch.tensor([[0.0]] * batch_size, requires_grad=True))

        result_tensor_sim = torch.zeros(
            (batch_size, self.length, self.n_classes))
        for i, result in enumerate(results):
            for j, r in enumerate(result):
                r = str(r).rjust(self.length, "0")
                result_prob = result_probs[i][j]
                for idx in range(self.length):
                    elt = int(r[idx])
                    result_tensor_sim[i][idx][elt] += result_prob

        result_tensor_sim = torch.nn.functional.normalize(
            result_tensor_sim, dim=2)

        # Vectorize the results
        _, result_tensor = super().vectorize(
            elements, element_indices, results, result_probs)

        # Return the elements mapping and also the result probability tensor
        return (elements, result_tensor_sim, result_tensor)

    def get_normalized_labels(self, y_pred, target, output_mapping):
        batch_size = y_pred.shape[0]
        y_sim = torch.zeros((batch_size, self.length, self.n_classes))
        for i, l in enumerate(target):
            l = str(l).rjust(self.length, "0")
            for idx in range(self.length):
                elt = int(l[idx])
                y_sim[i][idx][elt] = 1.0

        y = super().get_normalized_labels(y_pred, target, output_mapping)
        return y_sim, y


class StringOutputMapping(OutputMapping):
    def __init__(self, length, n_classes, fallback):
        self.length = length
        self.n_classes = n_classes
        if self.n_classes > 10:
            self.mapping = EMNIST_MAPPING
        else:
            self.mapping = [str(i) for i in range(self.n_classes)]
        self.labels_dict = self.get_labels_dict()
        self.fallback = fallback

    def dim(self):
        return self.n_classes ** self.length

    def generate_lists_rev_string(self, length):
        if length == 0:
            return [[]]  # Base case: an empty list
        else:
            smaller_lists = self.generate_lists_rev_string(length - 1)
            all_lists = []
            for lst in smaller_lists:
                for i in range(self.n_classes):
                    new_list = lst + [self.mapping[i]]
                    all_lists.append(new_list)
            return all_lists
        
    def generate_lists_common_prefix(self, length):
        all_lists = []
        for i in range(length, -1, -1):
            all_lists += self.generate_lists_rev_string(i)
        return all_lists

    def get_labels_dict(self):
        d = {}
        lsts = self.generate_lists_common_prefix(self.length)
        # lsts = self.generate_lists_rev_string(self.length)
        lsts = [''.join(lst) for lst in lsts]
        for i, lst in enumerate(lsts):
            d[tuple(lst)] = i
        return d

    def vectorize_label(self, labels):
        return torch.stack([
            torch.tensor([1.0 if self.labels_dict[tuple(label)] == e
                          else 0.0 for e in range(self.dim())])
            for label in labels])

class SudokuOutputMapping(OutputMapping):
    def __init__(self, size, fallback):
        self.size = size
        self.num_options = [str(i+1) for i in range(self.size)]
        self.fallback = fallback

    def vectorize(self, results: List, result_probs: torch.Tensor) -> torch.Tensor:
        batch_size, sample_count = result_probs.shape

        result_tensor_sim = torch.zeros(
            (batch_size, self.size, self.size, self.size))
        for i, result in enumerate(results):
            for j, r in enumerate(result):
                result_prob = result_probs[i][j]
                for row in range(self.size):
                    for col in range(self.size):
                        elt = r[row][col]
                        if elt in self.num_options:
                            idx = int(elt) - 1
                            result_tensor_sim[i][row][col][idx] += result_prob

        result_tensor_sim = torch.nn.functional.normalize(
            result_tensor_sim, dim=2)

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
        result_tensor = torch.nn.functional.normalize(
            result_tensor, dim=1)

        # Return the elements mapping and also the result probability tensor
        return (elements, result_tensor_sim, result_tensor)

    def get_normalized_labels(self, y_pred, target, output_mapping):
        batch_size = y_pred.shape[0]
        y_sim = torch.zeros((batch_size, self.size, self.size, self.size))
        for i, l in enumerate(target):
            for row in range(self.size):
                for col in range(self.size):
                    elt = l[row][col]
                    if elt in self.num_options:
                        idx = int(elt) - 1
                        y_sim[i][row][col][idx] = 1.0

        y = torch.tensor([1.0 if self.eval_result_eq(
            util.get_hashable_elem(l), m) else 0.0 for l in target for m in output_mapping]).view(batch_size, -1)

        return y_sim, y


class OutputMappingPermutations(OutputMapping):
    
    def __init__(self, rows, cols, dim, fallback):
        self.rows = rows
        self.cols = cols
        self.dim_om = dim
        self.labels_dict = self.get_labels_dict()
        self.fallback = fallback

    def dim(self):
        return self.dim_om

    def generate_lists(self, length):
        if length == 0:
            return [[]]  # Base case: an empty list
        else:
            smaller_lists = self.generate_lists(length - 1)
            all_lists = []
            for lst in smaller_lists:
                for i in range(10):
                    new_list = lst + [i]
                    all_lists.append(new_list)
            return all_lists

    def get_labels_dict(self):
        d = {}
        length = self.cols
        lsts = self.generate_lists(length)
        for i, lst in enumerate(lsts):
            d[tuple(lst)] = i
        return d

    def vectorize_label(self, labels):
        return torch.stack([
            torch.tensor([1.0 if self.labels_dict[tuple(label[0])] == e
                          else 0.0 for e in range(self.dim())])
            for label in labels])
        
class ListOutputMapping(OutputMapping):
    
    def __init__(self, length, n_classes, fallback):
        self.length = length
        self.n_classes = n_classes
        self.labels_dict = self.get_labels_dict()
        self.fallback = fallback

    def dim(self):
        return self.n_classes ** self.length

    def generate_lists(self, length):
        if length == 0:
            return [[]]
        else:
            smaller_lists = self.generate_lists(length - 1)
            all_lists = []
            for lst in smaller_lists:
                for i in range(self.n_classes):
                    new_list = lst + [i]
                    all_lists.append(new_list)
            return all_lists
        
    def generate_lists_all_lengths(self, length):
        all_lists = []
        for i in range(length, -1, -1):
            all_lists += self.generate_lists(i)
        return all_lists

    def get_labels_dict(self):
        d = {}
        lsts = self.generate_lists_all_lengths(self.length)
        for i, lst in enumerate(lsts):
            d[tuple(lst)] = i
        return d

    def vectorize_label(self, labels):
        return torch.stack([
            torch.tensor([1.0 if self.labels_dict[tuple(label)] == e
                          else 0.0 for e in range(self.dim())])
            for label in labels])

def get_output_mapping(output_config):
    om = output_config[OUTPUT_MAPPING]
    if om == UNKNOWN:
        return UnknownDiscreteOutputMapping(fallback=0)
    elif om == INT_OUTPUT_MAPPING:
        length = output_config[LENGTH]
        n_classes = output_config[N_CLASSES]
        return IntOutputMapping(length=length, n_classes=n_classes, fallback=0)
    elif om == "range":
        return RangeOutputMapping(output_config["low"], output_config["high"])
    elif om == LIST_OUTPUT_MAPPING:
        length = output_config[LENGTH]
        n_classes = output_config[N_CLASSES]
        return ListOutputMapping(length=length, n_classes=n_classes, fallback=0)
    elif om == STRING_OUTPUT_MAPPING:
        length = output_config[LENGTH]
        n_classes = output_config[N_CLASSES]
        return StringOutputMapping(length=length, n_classes=n_classes, fallback=0)
    elif om == DISCRETE_OUTPUT_MAPPING:
        if "range" in output_config:
            return DiscreteOutputMapping(elements=list(range(output_config["range"][0], output_config["range"][1])))
    elif om == OUTPUT_MAPPING_LIST_2D:
        dim = output_config[DIM]
        rows = output_config[N_ROWS]
        cols = output_config[N_COLS]
        return OutputMappingList2D(rows, cols, dim, fallback=0)
    elif om == PERMUTATIONS_OUTPUT_MAPPING:
        dim = output_config[DIM]
        rows = output_config[N_ROWS]
        cols = output_config[N_COLS]
        return OutputMappingPermutations(rows, cols, dim, fallback=0)
    elif om == SUDOKU_OUTPUT_MAPPING:
        size = output_config[N_ROWS]
        n_classes = output_config[N_CLASSES]
        return SudokuOutputMapping(size=size, fallback=0)
    else:
        raise Exception(f"Unknown output mapping {om}")
