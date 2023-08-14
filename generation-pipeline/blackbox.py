from typing import *
import torch
import errno
import os
import signal
import functools
import itertools
import random
from torch.multiprocessing import Pool

from constants import *
from input import *
from output import *


class TimeoutError(Exception):
    pass


# def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
#     def decorator(func):
#         def _handle_timeout(signum, frame):
#             raise TimeoutError(error_message)

#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             signal.signal(signal.SIGALRM, _handle_timeout)
#             signal.alarm(seconds)
#             try:
#                 result = func(*args, **kwargs)
#             finally:
#                 signal.alarm(0)
#             return result
#         return wrapper
#     return decorator


class BlackBoxFunction(torch.nn.Module):
    def __init__(
            self,
            function: Callable,
            input_mappings: Tuple[InputMapping],
            output_mapping: OutputMapping,
            batch_size: int,
            check_symmetry: bool = True,
            sample_count: int = 100,
            timeout_seconds: int = 1):
        super(BlackBoxFunction, self).__init__()
        assert type(input_mappings) == tuple, "input_mappings must be a tuple"
        self.function = function
        self.input_mappings = input_mappings
        self.output_mapping = output_mapping
        self.pool = Pool(processes=batch_size)
        self.sample_count = sample_count
        self.inputs_permute = True if check_symmetry and len(
            input_mappings) > 1 else False
        # self.timeout_decorator = timeout(seconds=timeout_seconds)

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
        input_permutations = self.get_permutations(sampled_indices, inputs)
        n_permutations = len([i for i in input_permutations])
        result_probs = torch.ones(
            (n_permutations, batch_size, self.sample_count))
        for i in range(len(sampled_indices)):
            input_tensor = inputs[i]
            input_permutations = self.get_permutations(sampled_indices, inputs)
            # individual_permutations = self.get_individual_permutations(
            #     sampled_indices[i], inputs)
            proofs = [(1 if perm[i] == i else 1) * input_tensor.gather(
                1, sampled_indices[perm[i]]) for perm in input_permutations]
            for (j, proof) in enumerate(proofs):
                result_probs[j] *= proof
        result_probs = torch.sum(result_probs, dim=0)

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
                y = self.function(*fn_input)
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

    def get_permutations(self, idxs, inputs):
        if self.inputs_permute and not random.randrange(0, 10):
            # 1/10 chance that we check whether inputs permute
            permutations = itertools.permutations(
                [i for i in range(len(idxs))])
            likely_inputs = [torch.argmax(input[0]).item() for input in inputs]
            fn_input = [self.input_mappings[i].combine(
                elt) for i, elt in enumerate(likely_inputs)]
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

    # def get_individual_permutations(self, sampled_indices, inputs):

    #     permutations = []
    #     for i, input in enumerate(inputs):
    #         p = self.input_mappings[i].permute(input)  # is it input or idx?
    #         permutations.append(p)
    #     return permutations

        # permutations = []
        # for i, input in enumerate(inputs):
        #     im = self.input_mappings[i]
        #     permutations.append(im.permute(input))
        #     pass
