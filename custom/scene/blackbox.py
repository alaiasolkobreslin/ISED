from typing import *
import torch

RESERVED_FAILURE = "__RESERVED_FAILURE__"

class ListInput:
    """
    The struct holding vectorized list input
    """
    def __init__(self, tensor: torch.Tensor, lengths: List[int]):
        self.tensor = tensor
        self.lengths = lengths

    def gather(self, dim: int, indices: torch.Tensor):
        batch_size = len(self.lengths)
        result = self.tensor.gather(dim, indices)
        
        # Reshape the sampled indices
        result = [result[self.lengths[:i].sum():self.lengths[:i+1].sum()].prod(dim=0) for i in range(batch_size)]
        result = torch.stack(result, dim=0)

        return result # 16 x 100


class InputMapping:
    def __init__(self): pass

    def sample(self, input: Any, sample_count: int, sample_strategy: str) -> Tuple[torch.Tensor, List[Any]]: pass


class ListInputMapping(InputMapping):
    def __init__(self, max_length: int, element_input_mapping: InputMapping):
        self.max_length = max_length
        self.element_input_mapping = element_input_mapping

    def sample(self, list_input: ListInput, sample_count: int, sample_strategy: str) -> Tuple[torch.Tensor, List[List[Any]]]:
        # Sample the elements
        batch_size = len(list_input.lengths) #list_input.tensor.shape[0], list_input.tensor.shape[1]
        # assert list_length == self.max_length, "inputs must have the same number of columns as the max length"
        # flattened = list_input.tensor.reshape((batch_size * list_length, -1))
        # flattened = [list_input.tensor[i, :list_input.lengths[i]] for i in range(batch_size)]
        # flattened = torch.cat(flattened, dim=0)
        flattened = list_input.tensor
        sampled_indices, sampled_elements = self.element_input_mapping.sample(flattened, sample_count, sample_strategy)

        # Reshape the sampled elements
        result_sampled_elements = []
        for i in range(batch_size):
            curr_elem = []
            for j in range(sample_count):
                curr_elem.append([i[j] for i in sampled_elements[list_input.lengths[:i].sum():list_input.lengths[:i+1].sum()]])
            result_sampled_elements.append(curr_elem)

        return (sampled_indices, result_sampled_elements)


class DiscreteInputMapping(InputMapping):
    def __init__(self, elements: List[Any]):
        self.elements = elements

    def sample(self, inputs: torch.Tensor, sample_count: int, sample_strategy: str) -> Tuple[torch.Tensor, List[Any]]:
        if sample_strategy == "categorical":
            num_input_elements = inputs.shape[1]
            assert num_input_elements == len(self.elements), "inputs must have the same number of columns as the number of elements"
            torch.distributions.Distribution.set_default_validate_args(False)
            distrs = torch.distributions.Categorical(probs=inputs)
            sampled_indices = distrs.sample((sample_count,)).transpose(0, 1)
            sampled_elements = [[self.elements[i] for i in sampled_indices_for_task_i] for sampled_indices_for_task_i in sampled_indices]
            return (sampled_indices, sampled_elements)
        elif sample_strategy == "top":
            num_input_elements = inputs.shape[1]
            assert num_input_elements == len(self.elements), "inputs must have the same number of columns as the number of elements"
            sampled_indices = torch.argmax(inputs)
            sampled_elements = [[self.elements[i] for i in sampled_indices_for_task_i] for sampled_indices_for_task_i in sampled_indices]
            return (sampled_indices, sampled_elements)


class OutputMapping:
    def __init__(self): pass

    def vectorize(self, results: List, result_probs: torch.Tensor, strategy: str):
        """
        An output mapping should implement this function to vectorize the results and result probabilities
        """
        pass


class DiscreteOutputMapping(OutputMapping):
    def __init__(self, elements: List[Any]):
        self.elements = elements
        self.element_indices = {e: i for (i, e) in enumerate(elements)}

    def vectorize(self, results: List, result_probs: torch.Tensor, strategy: str) -> torch.Tensor:
        batch_size, sample_count = result_probs.shape
        result_tensor = torch.zeros((batch_size, len(self.elements)))
        for i in range(batch_size):
            for j in range(sample_count):
                if results[i][j] != RESERVED_FAILURE:
                    if strategy == "minmax":
                        if result_probs[i, j] > result_tensor[i, self.element_indices[results[i][j]]]:
                            result_tensor[i, self.element_indices[results[i][j]]] = result_probs[i, j]
                    elif strategy == "addmult":
                        result_tensor[i, self.element_indices[results[i][j]]] += result_probs[i, j]
                    elif strategy == "maxmult":
                        if result_probs[i, j] > result_tensor[i, self.element_indices[results[i][j]]]:
                            result_tensor[i, self.element_indices[results[i][j]]] = result_probs[i, j]
        if strategy == "addmult":
            return torch.nn.functional.normalize(result_tensor, dim=1)
        else:
            return result_tensor


class UnknownDiscreteOutputMapping(OutputMapping):
    def __init__(self, fallback):
        self.fallback = fallback

    def vectorize(self, results: List[List], result_probs: torch.Tensor, strategy: str) -> torch.Tensor:
        batch_size, sample_count = result_probs.shape

        # Get the unique elements
        elements = list(set([elem for batch in results for elem in batch if elem != RESERVED_FAILURE]))
        element_indices = {e: i for (i, e) in enumerate(elements)}

        # If there is no element being derived...
        if len(elements) == 0:
            # We return a single fallback value, while the probability of result being fallback are all 0
            return ([self.fallback], torch.tensor([[0.0]] * batch_size, requires_grad=True))

        # Vectorize the results
        result_tensor = torch.zeros((batch_size, len(elements))).to("cpu")
        for i in range(batch_size):
            for j in range(sample_count):
                if results[i][j] != RESERVED_FAILURE:
                    if strategy == "minmax":
                        if result_probs[i, j] > result_tensor[i, element_indices[results[i][j]]]:
                            result_tensor[i, element_indices[results[i][j]]] = result_probs[i, j]
                    elif strategy == "addmult":
                        result_tensor[i, element_indices[results[i][j]]] += result_probs[i, j]
                    elif strategy == "maxmult":
                        if result_probs[i, j] > result_tensor[i, element_indices[results[i][j]]]:
                            result_tensor[i, element_indices[results[i][j]]] = result_probs[i, j]
                    else:
                        raise Exception(f"Unknown aggregation strategy {strategy}")
        if strategy == "addmult":
            result_tensor = torch.nn.functional.normalize(result_tensor, dim=1)

        # Return the elements mapping and also the result probability tensor
        return (elements, result_tensor)


class BlackBoxFunction(torch.nn.Module):
    def __init__(
            self,
            function: Callable,
            input_mappings: Tuple[InputMapping],
            output_mapping: OutputMapping,
            sample_count: int = 100,
            timeout_seconds: int = 1,
            sample_strategy: str = "categorical",
            aggregate_strategy: str = "addmult"):
        super(BlackBoxFunction, self).__init__()
        assert type(input_mappings) == tuple, "input_mappings must be a tuple"
        self.function = function
        self.input_mappings = input_mappings
        self.output_mapping = output_mapping
        self.sample_count = sample_count
        # self.timeout_decorator = timeout(seconds=timeout_seconds)
        self.sample_strategy = sample_strategy
        self.aggregate_strategy = aggregate_strategy

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
            sampled_indices_i, sampled_elements_i = input_mapping_i.sample(input_i, sample_count=self.sample_count, sample_strategy=self.sample_strategy)
            to_compute_inputs.append(sampled_elements_i)
            sampled_indices.append(sampled_indices_i)
        to_compute_inputs = self.zip_batched_inputs(to_compute_inputs)

        # Get the outputs from the black-box function
        results = self.invoke_function_on_batched_inputs(to_compute_inputs)

        # Aggregate the probabilities
        result_probs = torch.ones((batch_size, self.sample_count)).to("cpu")
        for (input_tensor, sampled_index) in zip(inputs, sampled_indices):
            if self.aggregate_strategy == "minmax":
                result_probs = torch.min(input_tensor.gather(1, sampled_index), result_probs)
            elif self.aggregate_strategy == "addmult":
                result_probs *= input_tensor.gather(1, sampled_index)
            elif self.aggregate_strategy == "maxmult":
                result_probs *= input_tensor.gather(1, sampled_index)

        # Vectorize the results back into a tensor
        return self.output_mapping.vectorize(results, result_probs, self.aggregate_strategy)

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
                y = self.function(*r)
                yield y
            except:
                yield RESERVED_FAILURE

    def invoke_function_on_batched_inputs(self, batched_inputs):
        return [list(self.invoke_function_on_inputs(batch)) for batch in batched_inputs]