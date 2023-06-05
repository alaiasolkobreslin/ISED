from typing import *

import torch


class ListInput:
    """
    The struct holding vectorized list input
    """

    def __init__(self, tensor: torch.Tensor, length: int):
        self.tensor = tensor
        self.length = length

    def gather(self, dim: int, indices: torch.Tensor):
        result = self.tensor.gather(dim + 1, indices)
        return torch.prod(result, dim=1)


class PaddedListInput:
    """
    The struct holding vectorized list input
    """

    def __init__(self, tensor: torch.Tensor, lengths: List[int]):
        self.tensor = tensor
        self.lengths = lengths

    def gather(self, dim: int, indices: torch.Tensor):
        result = self.tensor.gather(dim + 1, indices)
        return torch.prod(result, dim=1)


class InputMapping:
    def __init__(self): pass

    def sample(self, input: Any,
               sample_count: int) -> Tuple[torch.Tensor, List[Any]]: pass


class PaddedListInputMapping(InputMapping):
    def __init__(self, max_length: int, element_input_mapping: InputMapping, combine: Callable):
        self.max_length = max_length
        self.element_input_mapping = element_input_mapping
        self.combine = combine

    def sample(self, list_input: PaddedListInput, sample_count: int) -> Tuple[torch.Tensor, List[List[Any]]]:
        # Sample the elements
        batch_size, list_length = list_input.tensor.shape[0], list_input.tensor.shape[1]
        assert list_length == self.max_length, "inputs must have the same number of columns as the max length"
        flattened = list_input.tensor.reshape((batch_size * list_length, -1))
        sampled_indices, sampled_elements = self.element_input_mapping.sample(
            flattened, sample_count)

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


class ListInputMapping(InputMapping):
    def __init__(self, length: int, element_input_mapping: InputMapping, combine: Callable):
        self.length = length
        self.element_input_mapping = element_input_mapping
        self.combine = combine

    def sample(self, list_input: ListInput, sample_count: int) -> Tuple[torch.Tensor, List[List[Any]]]:
        # Sample the elements
        batch_size, list_length = list_input.tensor.shape[0], list_input.tensor.shape[1]
        assert list_length == self.length, "inputs must have the same number of columns as the length"
        flattened = list_input.tensor.reshape((batch_size * list_length, -1))
        sampled_indices, sampled_elements = self.element_input_mapping.sample(
            flattened, sample_count)

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


class DiscreteInputMapping(InputMapping):
    def __init__(self, elements: List[Any], combine: Callable):
        self.elements = elements
        self.combine = combine

    def sample(self, inputs: torch.Tensor, sample_count: int) -> Tuple[torch.Tensor, List[Any]]:
        num_input_elements = inputs.shape[1]
        assert num_input_elements == len(
            self.elements), "inputs must have the same number of columns as the number of elements"
        distrs = torch.distributions.Categorical(probs=inputs)
        sampled_indices = distrs.sample((sample_count,)).transpose(0, 1)
        sampled_elements = [[self.elements[i] for i in sampled_indices_for_task_i]
                            for sampled_indices_for_task_i in sampled_indices]
        return (sampled_indices, sampled_elements)
