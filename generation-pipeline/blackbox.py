from typing import *
import torch
import torch.nn.functional as F
import errno
import os
import signal
import functools
from torch.multiprocessing import Pool

from constants import *
from input import *
from output import *


class TimeoutError(Exception):
    pass


class BlackBoxFunction(torch.nn.Module):
    def __init__(
            self,
            function: Callable,
            input_mappings: Tuple[InputMapping],
            output_mapping: OutputMapping,
            batch_size: int,
            loss_aggregator: str,
            check_symmetry: bool = True,
            caching: bool = True,
            sample_count: int = 100,
            timeout_seconds: int = 1):
        super(BlackBoxFunction, self).__init__()
        assert type(input_mappings) == tuple, "input_mappings must be a tuple"
        self.function = function
        self.input_mappings = input_mappings
        self.output_mapping = output_mapping
        self.pool = Pool(processes=batch_size)
        self.loss_aggregator = loss_aggregator
        self.sample_count = sample_count
        self.caching = caching
        self.fn_cache = {}
        self.inputs_permute = True if check_symmetry and len(
            input_mappings) > 1 else False
        self.timeout_seconds = timeout_seconds
        self.timeout_decorator = self.decorator
        self.error_message = os.strerror(errno.ETIME)

    def decorator(self, func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(self.error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(self.timeout_seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper

    def forward(self, *inputs):
        num_inputs = len(inputs)
        assert num_inputs == len(
            self.input_mappings), "inputs and input_mappings must have the same length"

        # Get the batch size
        batch_size = inputs[0].batch_size()
        for i in range(1, num_inputs):
            assert batch_size == inputs[i].batch_size(
            ), "all inputs must have the same batch size"

        # Prepare the inputs to the black-box function
        to_compute_inputs, sampled_indices = [], []
        for (input_i, input_mapping_i) in zip(inputs, self.input_mappings):
            sampled_indices_i, sampled_elements_i = input_mapping_i.sample(
                input_i, sample_count=self.sample_count)
            input_for_pooling = input_i.get_input_for_pooling()
            if input_for_pooling:
                to_compute = [[(s, input_for_pooling[idx]) for s in sampled_element]
                              for idx, sampled_element in enumerate(sampled_elements_i)]
            else:
                to_compute = sampled_elements_i
            to_compute_inputs.append(to_compute)
            sampled_indices.append(sampled_indices_i)
        to_compute_inputs = self.zip_batched_inputs(to_compute_inputs)

        # Get the outputs from the black-box function
        results = self.invoke_function_on_batched_inputs(to_compute_inputs)

        # Aggregate the probabilities
        result_probs = torch.ones((batch_size, self.sample_count), device=DEVICE)
        for (input_tensor, sampled_index) in zip(inputs, sampled_indices):
            gathered_probs = input_tensor.gather(1, sampled_index)
            if self.loss_aggregator == ADD_MULT:
                result_probs *= gathered_probs
            elif self.loss_aggregator == MIN_MAX:
                result_probs = torch.minimum(
                    result_probs.clone(), gathered_probs)

        # Vectorize the results back into a tensor
        return self.output_mapping.vectorize(results, result_probs)

    def zip_batched_inputs(self, batched_inputs):
        result = [list(zip(*lists)) for lists in zip(*batched_inputs)]
        return result

    def invoke_function_on_inputs(self, input_args):
        """
        Given a list of inputs, invoke the black-box function on each of them.
        Note that function may fail on some inputs, and we skip those.
        """
        for r in input_args:
            try:
                fn_input = (self.input_mappings[i].combine(
                    elt) for i, elt in enumerate(r))
                if not self.caching:
                    yield self.function(*fn_input)
                else:
                    hashable_fn_input = util.get_hashable_elem(fn_input)
                    if hashable_fn_input in self.fn_cache:
                        yield self.fn_cache[hashable_fn_input]
                    else:
                        y = self.timeout_decorator(self.function)(*fn_input)
                        self.fn_cache[hashable_fn_input] = y
                        yield y
            except:
                yield RESERVED_FAILURE

    def process_batch(self, batch):
        return list(self.invoke_function_on_inputs(batch))

    def invoke_function_on_batched_inputs(self, batched_inputs):
        return self.pool.map(self.process_batch, batched_inputs)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
