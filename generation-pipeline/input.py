import itertools
from typing import List, Tuple
import torch

from constants import *

from typing import *


class Input:

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def batch_size(self):
        return self.tensor.shape[0]

    def gather(self, dim: int, indices: torch.Tensor):
        pass

    def gather_permutations(self, dim: int, indices: torch.Tensor, permutations: List[Tuple]):
        pass

    def get_input_for_pooling(self):
        pass


class SingleInput(Input):
    def __init__(self, tensor: torch.Tensor):
        super(SingleInput, self).__init__(tensor)

    def gather(self, dim: int, indices: torch.Tensor):
        return self.tensor.gather(dim, indices)

    def gather_permutations(self, dim: int, indices: torch.Tensor, permutations: List[Tuple]):
        return self.tensor.gather(dim, indices)


class ListInput(Input):
    """
    The struct holding vectorized list input
    """

    def __init__(self, tensor: torch.Tensor, length: int):
        super(ListInput, self).__init__(tensor)
        self.length = length

    def gather(self, dim: int, indices: torch.Tensor):
        result = self.tensor.gather(dim + 1, indices)
        return torch.prod(result, dim=1)

    def gather_permutations(self, dim: int, indices: torch.Tensor, permutations: List[Tuple]):
        _, length, _ = indices.shape
        proofs = []
        tensor_lst = [i for i in torch.transpose(self.tensor, 0, 1)]
        for perm in permutations:
            permuted = torch.stack([tensor_lst[perm[i]]
                                   for i in range(length)])
            new_tensor = torch.transpose(permuted, 0, 1)
            new_tensor_gathered = new_tensor.gather(dim+1, indices)
            proofs.append(torch.prod(new_tensor_gathered, dim=1))
        return proofs


class PaddedListInput(Input):
    """
    The struct holding vectorized list input
    """

    def __init__(self, tensor: torch.Tensor, lengths: List[int]):
        super(PaddedListInput, self).__init__(tensor)
        self.lengths = lengths

    def gather(self, dim: int, indices: torch.Tensor):
        result = self.tensor.gather(dim + 1, indices)
        return torch.prod(result, dim=1)


class PaddedListInputSudoku(Input):
    """
    The struct holding vectorized list input
    """

    def __init__(self, tensor: torch.Tensor, lengths: List[int], bool_boards=torch.Tensor):
        super(PaddedListInputSudoku, self).__init__(tensor)
        self.lengths = lengths
        self.bool_boards = bool_boards

    def gather(self, dim: int, indices: torch.Tensor):
        result = self.tensor.gather(dim + 1, indices)
        return torch.prod(result, dim=1)

    def get_input_for_pooling(self):
        return [b for b in self.bool_boards]


class ListInput2D(Input):

    def __init__(self, tensor: torch.Tensor, n_rows: int, n_cols: int):
        super(ListInput2D, self).__init__(tensor)
        self.n_rows = n_rows
        self.n_cols = n_cols

    def gather(self, dim: int, indices: torch.Tensor):
        result = self.tensor.gather(dim + 2, indices)
        return torch.prod(torch.prod(result, dim=1), dim=1)

    def gather_permutations(self, dim: int, indices: torch.Tensor, permutations: List[Tuple]):
        batch_size, rows, cols, _ = indices.shape
        proofs = []
        once_transposed = torch.transpose(self.tensor, 0, 1)
        twice_transposed = torch.transpose(once_transposed, 1, 2)
        tensor_lst = [i for j in twice_transposed for i in j]
        for perm in permutations:
            permuted = torch.stack([tensor_lst[perm[i]]
                                   for i in range(rows * cols)]).view(rows, cols, batch_size, -1)
            new_tensor_transposed = torch.transpose(permuted, 1, 2)
            new_tensor = torch.transpose(new_tensor_transposed, 0, 1)
            new_tensor_gathered = new_tensor.gather(dim+2, indices)
            proofs.append(torch.prod(torch.prod(
                new_tensor_gathered, dim=1), dim=1))
        return proofs

class InputMapping:
    def __init__(self): pass

    def shape(self): pass

    def sample(self, input: Any,
               sample_count: int) -> Tuple[torch.Tensor, List[Any]]: pass

    def argmax(self, input: Any) -> Tuple[torch.Tensor, List[Any]]: pass


class PaddedListInputMapping(InputMapping):
    def __init__(self, max_length: int, element_input_mapping: InputMapping, combine: Callable):
        self.max_length = max_length
        self.element_input_mapping = element_input_mapping
        self.combine = combine
        self.does_permute = True

    def sample(self, list_input: PaddedListInput, sample_count: int) -> Tuple[torch.Tensor, List[List[Any]]]:
        # Sample the elements
        batch_size, list_length = list_input.tensor.shape[0], list_input.tensor.shape[1]
        assert list_length == self.max_length, "inputs must have the same number of columns as the max length"
        flattened = list_input.tensor.reshape((batch_size * list_length, -1))
        sampled_indices, sampled_elements = self.element_input_mapping.sample(
            SingleInput(flattened), sample_count)

        # Reshape the sampled elements
        result_sampled_elements = []
        for i in range(batch_size):
            curr_batch = []
            for j in range(sample_count):
                curr_elem = []
                for k in range(list_input.lengths[i]):
                    curr_elem.append(sampled_elements[i * list_length + k][j])
                curr_batch.append(curr_elem)
            result_sampled_elements.append(curr_batch)

        # Reshape the sampled indices
        sampled_indices_original_shape = tuple(sampled_indices.shape[1:])
        sampled_indices = sampled_indices.reshape(
            batch_size, list_length, *sampled_indices_original_shape)

        return (sampled_indices, result_sampled_elements)

    def argmax(self, input: PaddedListInput):
        pass


class ListInputMapping(InputMapping):
    def __init__(self, length: int, element_input_mapping: InputMapping, combine: Callable):
        self.length = length
        self.element_input_mapping = element_input_mapping
        self.combine = combine
        self.does_permute = True

    def sample(self, list_input: ListInput, sample_count: int) -> Tuple[torch.Tensor, List[List[Any]]]:
        # Sample the elements
        batch_size, list_length = list_input.tensor.shape[0], list_input.tensor.shape[1]
        assert list_length == self.length, "inputs must have the same number of columns as the length"
        flattened = list_input.tensor.reshape((batch_size * list_length, -1))
        sampled_indices, sampled_elements = self.element_input_mapping.sample(
            SingleInput(flattened), sample_count)

        # Reshape the sampled elements
        result_sampled_elements = []
        for i in range(batch_size):
            curr_batch = []
            for j in range(sample_count):
                curr_elem = []
                for k in range(list_input.length):
                    curr_elem.append(sampled_elements[i * list_length + k][j])
                curr_batch.append(curr_elem)
            result_sampled_elements.append(curr_batch)

        # Reshape the sampled indices
        sampled_indices_original_shape = tuple(sampled_indices.shape[1:])
        sampled_indices = sampled_indices.reshape(
            batch_size, list_length, *sampled_indices_original_shape)

        return (sampled_indices, result_sampled_elements)

    def argmax(self, input: ListInput):
        pass


class ListInputMapping2D(InputMapping):
    def __init__(self, n_rows: int, n_cols, element_input_mapping: InputMapping, combine: Callable):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.element_input_mapping = element_input_mapping
        self.combine = combine
        self.does_permute = True

    def sample(self, list_input: ListInput2D, sample_count: int) -> Tuple[torch.Tensor, List[List[Any]]]:
        # Sample the elements
        batch_size = list_input.tensor.shape[0]
        n_rows = list_input.tensor.shape[1]
        n_cols = list_input.tensor.shape[2]
        assert (n_rows == self.n_rows and n_cols ==
                self.n_cols), "inputs dimensions must match n_rows and n_cols"
        flattened = list_input.tensor.reshape(
            (batch_size * n_rows * n_cols, -1))
        sampled_indices, sampled_elements = self.element_input_mapping.sample(
            SingleInput(flattened), sample_count)

        # Reshape the sampled elements
        result_sampled_elements = []
        for i in range(batch_size):
            curr_batch = []
            for j in range(sample_count):
                curr_elem = []
                for k in range(list_input.n_rows * list_input.n_cols):
                    curr_elem.append(
                        sampled_elements[i * n_rows * n_cols + k][j])
                curr_batch.append(curr_elem)
            result_sampled_elements.append(curr_batch)

        # Reshape the sampled indices
        sampled_indices_original_shape = tuple(sampled_indices.shape[1:])
        sampled_indices = sampled_indices.reshape(
            batch_size, n_rows, n_cols, *sampled_indices_original_shape)

        return (sampled_indices, result_sampled_elements)

    def argmax(self, input: ListInput2D):
        pass


class DiscreteInputMapping(InputMapping):
    def __init__(self, elements: List[Any], combine: Callable):
        self.elements = elements
        self.combine = combine

    def shape(self):
        return (len(self.elements),)

    def sample(self, inputs: SingleInput, sample_count: int) -> Tuple[torch.Tensor, List[Any]]:
        num_input_elements = inputs.tensor.shape[1]
        assert num_input_elements == len(
            self.elements), "inputs must have the same number of columns as the number of elements"
        distrs = torch.distributions.Categorical(probs=inputs.tensor)
        sampled_indices = distrs.sample((sample_count,)).transpose(0, 1)
        sampled_elements = [[self.elements[i] for i in sampled_indices_for_task_i]
                            for sampled_indices_for_task_i in sampled_indices]
        return (sampled_indices, sampled_elements)

    def argmax(self, inputs: SingleInput):
        num_input_elements = inputs.tensor.shape[1]
        assert num_input_elements == len(
            self.elements), "inputs must have the same number of columns as the number of elements"
        max_indices = torch.argmax(inputs.tensor, dim=1)
        max_elements = [self.elements[i] for i in max_indices]
        return (max_indices, max_elements)

