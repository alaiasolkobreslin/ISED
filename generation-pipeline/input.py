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


class CoffeeInput(Input):
    def __init__(self, tensor: torch.Tensor, lengths: List[int], areas: List[List[int]]):
        super(CoffeeInput, self).__init__(tensor)
        self.lengths = lengths
        self.areas = areas

    def gather(self, dim: int, indices: torch.Tensor):
        result = self.tensor.gather(dim + 1, indices)
        return torch.prod(result, dim=1)


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

        # # _, _, samples = indices.shape
        # result = self.tensor.gather(dim + 1, indices)
        # final_results = []
        # for i, batch in enumerate(result):
        #     length_i = self.lengths[i]
        #     # collected = batch.view(length_i, samples)
        #     collected = batch[:length_i]
        #     final_results.append(torch.prod(collected, dim=0))
        # return torch.stack(final_results)
        # # result = self.tensor.gather(dim + 1, indices)
        # # return torch.prod(result, dim=1)


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


class ListInput2DSudoku(Input):

    def __init__(self, tensor: torch.Tensor, selected: torch.Tensor, n_rows: int, n_cols: int):
        super(ListInput2DSudoku, self).__init__(tensor)
        self.selected = selected
        self.n_rows = n_rows
        self.n_cols = n_cols

    def gather(self, dim: int, indices: torch.Tensor):
        _, _, _, samples = indices.shape
        result = self.tensor.gather(dim + 2, indices)
        final_results = []
        for i, batch in enumerate(result):
            selected_i = self.selected[i]
            collected = torch.ones(samples, device=DEVICE)
            for j, row in enumerate(selected_i):
                for k, col in enumerate(row):
                    if col:
                        collected *= batch[j][k]
            final_results.append(collected)
        return torch.stack(final_results)


class VideoInput(Input):
    """
    The struct holding vectorized list input
    """

    def __init__(self, tensor: torch.Tensor, change: torch.Tensor, length: int):
        super(VideoInput, self).__init__(tensor)
        self.change = change
        self.length = length

    def gather(self, dim: int, indices: torch.Tensor):
        result = self.tensor.gather(dim + 1, indices)
        return torch.prod(result, dim=1)


class InputMapping:
    def __init__(self): pass

    def shape(self): pass

    def sample(self, input: Any,
               sample_count: int) -> Tuple[torch.Tensor, List[Any]]: pass

    def argmax(self, input: Any) -> Tuple[torch.Tensor, List[Any]]: pass

    # def permute(self) -> List[Any]: pass


class PaddedListInputMappingCoffee(InputMapping):
    def __init__(self, max_length: int, element_input_mapping: InputMapping, combine: Callable):
        self.max_length = max_length
        self.element_input_mapping = element_input_mapping
        self.combine = combine
        self.does_permute = True

    def sample(self, list_input: CoffeeInput, sample_count: int) -> Tuple[torch.Tensor, List[List[Any]]]:
        # Sample the elements
        batch_size, list_length = list_input.tensor.shape[0], list_input.tensor.shape[1]
        assert list_length == self.max_length, "inputs must have the same number of columns as the max length"
        flattened = list_input.tensor.reshape((batch_size * list_length, -1))
        sampled_indices, sampled_elements = self.element_input_mapping.sample(
            SingleInput(flattened), sample_count)

        # Reshape the sampled elements
        result_sampled_elements = []
        for i in range(batch_size):
            curr_batch_selected = []
            for j in range(sample_count):
                curr_elem_selected = []
                for k in range(list_input.lengths[i]):
                    curr_elem_selected.append((
                        sampled_elements[i * list_length + k][j], list_input.areas[i][k]))
                curr_batch_selected.append(curr_elem_selected)
            result_sampled_elements.append(curr_batch_selected)

        # Reshape the sampled indices
        sampled_indices_original_shape = tuple(sampled_indices.shape[1:])
        sampled_indices = sampled_indices.reshape(
            batch_size, list_length, *sampled_indices_original_shape)

        return (sampled_indices, result_sampled_elements)

    def argmax(self, list_input: CoffeeInput):
        pass


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

    # def permute(self):
    #     idx_lst = [i for i in range(self.max_length)]
    #     if not self.does_permute:
    #         return [idx_lst]
    #     return [p for p in itertools.permutations(idx_lst)]


class PaddedListInputMappingSudoku(InputMapping):
    def __init__(self, max_length: int, element_input_mapping: InputMapping, combine: Callable):
        self.max_length = max_length
        self.element_input_mapping = element_input_mapping
        self.combine = combine
        self.does_permute = True

    def sample(self, list_input: PaddedListInputSudoku, sample_count: int) -> Tuple[torch.Tensor, List[List[Any]]]:
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

    def argmax(self, input: PaddedListInputSudoku):
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

    # def permute(self):
    #     idx_lst = [i for i in range(self.length)]
    #     if not self.does_permute:
    #         return [idx_lst]
    #     return [p for p in itertools.permutations(idx_lst)]


class ListInputMapping2DSudoku(InputMapping):
    def __init__(self, n_rows: int, n_cols: int, max_length: int, element_input_mapping: InputMapping, combine: Callable):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.max_length = max_length
        self.element_input_mapping = element_input_mapping
        self.combine = combine
        self.does_permute = True

    def sample(self, list_input: ListInput2DSudoku, sample_count: int) -> Tuple[torch.Tensor, List[List[Any]]]:
        # Sample the elements
        batch_size = list_input.tensor.shape[0]
        length = list_input.tensor.shape[1]
        assert (length == self.max_length), "inputs dimensions must match max length"
        flattened = list_input.tensor.reshape(
            (batch_size * length, -1))
        sampled_indices, sampled_elements = self.element_input_mapping.sample(
            SingleInput(flattened), sample_count)

        # Reshape the sampled elements
        result_sampled_elements = []
        for i in range(batch_size):
            curr_batch = []
            for j in range(sample_count):
                curr_elem = []
                for k in range(list_input.n_rows * list_input.n_cols):
                    row = k // self.n_rows
                    col = k % self.n_cols
                    selected = list_input.selected[i][row][col].item()
                    if selected:
                        curr_elem.append(
                            str(sampled_elements[i * self.n_rows * self.n_cols + k][j]))
                    else:
                        curr_elem.append('.')
                curr_batch.append(curr_elem)
            result_sampled_elements.append(curr_batch)

        # Reshape the sampled indices
        sampled_indices_original_shape = tuple(sampled_indices.shape[1:])
        sampled_indices = sampled_indices.reshape(
            batch_size, self.n_rows, self.n_cols, *sampled_indices_original_shape)

        return (sampled_indices, result_sampled_elements)

    def argmax(self, input: ListInput2DSudoku):
        pass

    # def permute(self):
    #     return []


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

    # def permute(self):
    #     all_idxs = [i for i in range(self.n_rows * self.n_cols)]
    #     if not self.does_permute:
    #         return [all_idxs]
    #     permutations = itertools.permutations(all_idxs)
    #     return [p for p in permutations]


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

    # def permute(self):
    #     return []
