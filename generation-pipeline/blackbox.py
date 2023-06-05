from typing import *
import torch
import errno
import os
import signal
import functools

from constants import *
from input import *
from output import *


class TimeoutError(Exception):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator


class BlackBoxFunction(torch.nn.Module):
    def __init__(
            self,
            function: Callable,
            input_mappings: Tuple[InputMapping],
            output_mapping: OutputMapping,
            sample_count: int = 100,
            timeout_seconds: int = 1):
        super(BlackBoxFunction, self).__init__()
        assert type(input_mappings) == tuple, "input_mappings must be a tuple"
        self.function = function
        self.input_mappings = input_mappings
        self.output_mapping = output_mapping
        self.sample_count = sample_count
        self.timeout_decorator = timeout(seconds=timeout_seconds)

    def forward(self, *inputs):
        num_inputs = len(inputs)
        assert num_inputs == len(
            self.input_mappings), "inputs and input_mappings must have the same length"

        # Get the batch size
        batch_size = self.get_batch_size(inputs[0])
        for i in range(1, num_inputs):
            assert batch_size == self.get_batch_size(
                inputs[i]), "all inputs must have the same batch size"

        # Prepare the inputs to the black-box function
        to_compute_inputs, sampled_indices = [], []
        for (input_i, input_mapping_i) in zip(inputs, self.input_mappings):
            sampled_indices_i, sampled_elements_i = input_mapping_i.sample(
                input_i, sample_count=self.sample_count)
            to_compute_inputs.append(sampled_elements_i)
            sampled_indices.append(sampled_indices_i)
        to_compute_inputs = self.zip_batched_inputs(to_compute_inputs)

        # Get the outputs from the black-box function
        results = self.invoke_function_on_batched_inputs(to_compute_inputs)

        # Aggregate the probabilities
        result_probs = torch.ones((batch_size, self.sample_count))
        for (input_tensor, sampled_index) in zip(inputs, sampled_indices):
            result_probs *= input_tensor.gather(1, sampled_index)

        # Vectorize the results back into a tensor
        return self.output_mapping.vectorize(results, result_probs)

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
                y = self.timeout_decorator(self.function)(*r)
                yield y
            except:
                yield RESERVED_FAILURE

    def invoke_function_on_batched_inputs(self, batched_inputs):
        return [list(self.invoke_function_on_inputs(batch)) for batch in batched_inputs]
