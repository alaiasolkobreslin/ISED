from typing import *
import random
import torch
import itertools


RESERVED_FAILURE = "__RESERVED_FAILURE__"


class ListInput:
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

    def sample(self, input: Any, sample_count: int) -> Tuple[torch.Tensor, List[Any]]: pass


class ListInputMapping(InputMapping):
    def __init__(self, max_length: int, element_input_mapping: InputMapping):
        self.max_length = max_length
        self.element_input_mapping = element_input_mapping

    def sample(self, list_input: ListInput, sample_count: int) -> Tuple[torch.Tensor, List[List[Any]]]:
        # Sample the elements
        batch_size, list_length = list_input.tensor.shape[0], list_input.tensor.shape[1]
        assert list_length == self.max_length, "inputs must have the same number of columns as the max length"
        flattened = list_input.tensor.reshape((batch_size * list_length, -1))
        sampled_indices, sampled_elements = self.element_input_mapping.sample(flattened, sample_count)

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
        sampled_indices = sampled_indices.reshape(batch_size, list_length, *sampled_indices_original_shape)

        return (sampled_indices, result_sampled_elements)


class DiscreteInputMapping(InputMapping):
    def __init__(self, elements: List[Any]):
        self.elements = elements

    def sample(self, inputs: torch.Tensor, sample_count: int) -> Tuple[torch.Tensor, List[Any]]:
        num_input_elements = inputs.shape[1]
        assert num_input_elements == len(self.elements), "inputs must have the same number of columns as the number of elements"
        distrs = torch.distributions.Categorical(probs=inputs)
        sampled_indices = distrs.sample((sample_count,)).transpose(0, 1)
        sampled_elements = [[self.elements[i] for i in sampled_indices_for_task_i] for sampled_indices_for_task_i in sampled_indices]
        return (sampled_indices, sampled_elements)


class OutputMapping:
    def __init__(self): pass

    def vectorize(self, loss_aggregator, results: List, result_probs: torch.Tensor):
        """
        An output mapping should implement this function to vectorize the results and result probabilities
        """
        pass


class DiscreteOutputMapping(OutputMapping):
    def __init__(self, elements: List[Any]):
        self.elements = elements
        self.element_indices = {e: i for (i, e) in enumerate(elements)}

    def vectorize(self, loss_aggregator, results: List, result_probs: torch.Tensor) -> torch.Tensor:
        batch_size, sample_count = result_probs.shape
        result_tensor = torch.zeros((batch_size, len(self.elements)))
        for i in range(batch_size):
            for j in range(sample_count):
                # print(results[i][j])
                if results[i][j] != RESERVED_FAILURE:
                    if loss_aggregator == 'min_max':
                        result_tensor[i, self.element_indices[results[i][j]]] = torch.max(result_tensor[i, self.element_indices[results[i][j]]].clone(), result_probs[i, j])
                    elif loss_aggregator == 'add_mult':
                        result_tensor[i, self.element_indices[results[i][j]]] += result_probs[i, j]
                    else:
                        raise Exception(f"Unknown loss aggregator: {self.loss_aggregator}")
        return torch.nn.functional.normalize(result_tensor, dim=1)


class UnknownDiscreteOutputMapping(OutputMapping):
    def __init__(self):
        pass

    def vectorize(self, loss_aggregator, results: List, result_probs: torch.Tensor) -> torch.Tensor:
        batch_size, sample_count = result_probs.shape

        # Get the unique elements
        elements = list(set([elem for batch in results for elem in batch if elem != RESERVED_FAILURE]))
        element_indices = {e: i for (i, e) in enumerate(elements)}

        # Vectorize the results
        result_tensor = torch.zeros((batch_size, len(elements)))
        for i in range(batch_size):
            for j in range(sample_count):
                if results[i][j] != RESERVED_FAILURE:
                    if loss_aggregator == 'min_max':
                        result_tensor[i, element_indices[results[i][j]]] = torch.max(result_tensor[i, element_indices[results[i][j]]].clone(), result_probs[i, j])
                    elif loss_aggregator == 'add_mult':
                        result_tensor[i, element_indices[results[i][j]]] += result_probs[i, j]
                    else:
                        raise Exception(f"Unknown loss aggregator: {self.loss_aggregator}")
        result_tensor = torch.nn.functional.normalize(result_tensor, dim=1)

        # Return the elements mapping and also the result probability tensor
        return (elements, result_tensor)


class BlackBoxFunction(torch.nn.Module):
    def __init__(
            self,
            function: Callable,
            input_mappings: Tuple[InputMapping],
            output_mapping: OutputMapping,
            caching: bool = True,
            loss_aggregator: str = "add_mult",
            sample_count: int = 100):
        super(BlackBoxFunction, self).__init__()
        assert type(input_mappings) == tuple, "input_mappings must be a tuple"
        self.function = function
        self.input_mappings = input_mappings
        self.output_mapping = output_mapping
        self.sample_count = sample_count
        self.loss_aggregator = loss_aggregator
        self.inputs_permute = False
        self.caching = caching
        self.cache = {}

    def forward(self, *inputs):
        num_inputs = len(inputs)
        assert num_inputs == len(self.input_mappings), "inputs and input_mappings must have the same length"

        # Get the batch size
        batch_size = self.get_batch_size(inputs[0])
        for i in range(1, num_inputs):
            assert batch_size == self.get_batch_size(inputs[i]), "all inputs must have the same batch size"

        # Prepare the inputs to the black-box function
        to_compute_inputs, sampled_indices = [], []
        for (input_i, input_mapping_i) in zip(inputs, self.input_mappings):
            sampled_indices_i, sampled_elements_i = input_mapping_i.sample(input_i, sample_count=self.sample_count)
            to_compute_inputs.append(sampled_elements_i)
            sampled_indices.append(sampled_indices_i)
        to_compute_inputs = self.zip_batched_inputs(to_compute_inputs)

        # Get the outputs from the black-box function
        results = self.invoke_function_on_batched_inputs(to_compute_inputs)

        # Aggregate the probabilities
        input_permutations = self.get_permutations(sampled_indices, inputs)
        n_permutations = len([i for i in input_permutations])
        result_probs = torch.ones(
            (n_permutations, batch_size, self.sample_count))
        for i in range(len(sampled_indices)):
            input_tensor = inputs[i]
            input_permutations = self.get_permutations(sampled_indices, inputs)
            proofs = [(1 if perm[i] == i else 1) * input_tensor.gather(1, sampled_indices[perm[i]]) for perm in input_permutations]
            for (j, proof) in enumerate(proofs):
                if self.loss_aggregator == "add_mult":
                    result_probs[j] *= proof
                elif self.loss_aggregator == "min_max":
                    result_probs[j] = torch.minimum(
                        result_probs[j].clone(), proof)
                else:
                    raise Exception(
                        f"Unknown loss aggregator: {self.loss_aggregator}")
        result_probs = torch.sum(result_probs, dim=0)

        # Vectorize the results back into a tensor
        return self.output_mapping.vectorize(self.loss_aggregator, results, result_probs)

    def get_batch_size(self, input: Any):
        if type(input) == torch.Tensor:
            return input.shape[0]
        elif type(input) == ListInput:
            return len(input.lengths)
        raise Exception("Unknown input type")

    def zip_batched_inputs(self, batched_inputs):
        result = [list(zip(*lists)) for lists in zip(*batched_inputs)]
        return result

    def invoke_function_on_inputs(self, inputs):
        """
        Given a list of inputs, invoke the black-box function on each of them.
        Note that function may fail on some inputs, and we skip those.
        """
        for r in inputs:
            try:
                if self.caching and r in self.cache:
                    y = self.cache[r]
                else:
                    y = self.function(*r)
                    if self.caching:
                        self.cache[r] = y
                yield y
            except:
                yield RESERVED_FAILURE

    def invoke_function_on_batched_inputs(self, batched_inputs):
        return [list(self.invoke_function_on_inputs(batch)) for batch in batched_inputs]

    def get_permutations(self, idxs, inputs):
        if self.inputs_permute and not random.randrange(0, 10):
            # 1/10 chance that we check whether inputs permute
            permutations = itertools.permutations(
                [i for i in range(len(idxs))])
            likely_inputs = [torch.argmax(input[0]).item() for input in inputs]
            fn_input = [id(elt) for i, elt in enumerate(likely_inputs)]
            original_output = self.function(*fn_input)
            for perm in permutations:
                permuted_inputs = [fn_input[i] for i in perm]
                new_output = self.function(*permuted_inputs)
                if new_output != original_output:
                    self.inputs_permute = False
        if self.inputs_permute:
            permutations = itertools.permutations(
                [i for i in range(len(idxs))])
            permutations_list = [p for p in permutations]
            # TODO: This cutoff size is hardcoded. Fix this?
            if len(permutations_list) > 120:
                permutations_list = permutations_list[:120]
            return permutations_list
        else:
            permutations = [i for i in range(len(idxs))]
            return [permutations]