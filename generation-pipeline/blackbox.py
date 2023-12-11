from typing import *
import torch
import torch.nn.functional as F
import errno
import os
import signal
import functools
from functools import partial
import itertools
import random
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
        input_permutations = self.get_permutations(sampled_indices, inputs)
        n_permutations = len([i for i in input_permutations])
        result_probs = torch.ones(
            (n_permutations, batch_size, self.sample_count))
        for i in range(len(sampled_indices)):
            input_tensor = inputs[i]
            input_permutations = self.get_permutations(sampled_indices, inputs)
            proofs = [(1 if perm[i] == i else 1) * input_tensor.gather(
                1, sampled_indices[perm[i]]) for perm in input_permutations]
            for (j, proof) in enumerate(proofs):
                if self.loss_aggregator == ADD_MULT:
                    result_probs[j] *= proof
                elif self.loss_aggregator == MIN_MAX:
                    result_probs[j] = torch.minimum(
                        result_probs[j].clone(), proof)
                else:
                    raise Exception(
                        f"Unknown loss aggregator: {self.loss_aggregator}")
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


class BlackBoxFunctionFiniteDifference(torch.autograd.Function):

    bbox = None

    def compute_prob(argmax, inputs):
        n = len(argmax)
        batch_size = argmax[0].shape[0]
        probs, freqs = torch.zeros(n, batch_size, batch_size, 1), torch.zeros(
            n, batch_size, batch_size)
        for i in range(n):
            freq = torch.zeros(batch_size, batch_size)
            prob = torch.ones(batch_size, batch_size, 1)
            for j in range(batch_size):
                input_i = argmax[i][j]
                prob[j] = inputs[i][:, input_i].unsqueeze(1)
                freq[j] = torch.where(argmax[i] == input_i, 1, 0)
            probs[i] = prob
            freqs[i] = freq
        return probs, freqs

    def finite_difference(fn, *inputs):
        argmax_inputs, inputs_distr, argmax_x = [], [], []
        k = 0
        for x in inputs:
            n = x.shape[1]
            k += n
            argmax_x.append(x.argmax(dim=1))
            argmax_inputs.append(F.one_hot(x.argmax(dim=1), num_classes=n))
            inputs_distr.append(x)

        y_pred = fn(*tuple(argmax_inputs))
        probs, freqs = BlackBoxFunctionFiniteDifference.compute_prob(
            argmax_x, inputs_distr)

        jacobian = []
        for x, count in zip(inputs, range(len(inputs))):
            batch_size, n = x.shape
            jacobian_i = torch.zeros(batch_size, k, n)
            probs_c = torch.cat((probs[:count], probs[count+1:]))
            freqs_c = torch.cat((freqs[:count], freqs[count+1:]))
            for i in torch.arange(0, n):
                inputs_i = argmax_inputs.copy()
                inputs_i[count] = F.one_hot(
                    i, num_classes=n).float().repeat(batch_size, 1)
                y = fn(*tuple(inputs_i)) - y_pred
                for batch_i in range(batch_size):
                    probs_i = probs_c[:, batch_i, :].prod(dim=0)
                    freqs_i = freqs_c[:, batch_i, :].prod(dim=0).sum()
                    jacobian_i[:, :, i] += y[batch_i].unsqueeze(
                        0).repeat(batch_size, 1)*probs_i/(n*freqs_i)
            jacobian.append(jacobian_i)
        return tuple(jacobian)

    def zip_batched_inputs(batched_inputs):
        return list(zip(*batched_inputs))

    def invoke_function_on_inputs(bbox, r):
        """
        Given a list of inputs, invoke the black-box function on each of them.
        Note that function may fail on some inputs, and we skip those.
        """
        try:
            fn_input = (bbox.input_mappings[i].combine(
                elt) for i, elt in enumerate(r))
            if not bbox.caching:
                yield bbox.function(*fn_input)
            else:
                hashable_fn_input = util.get_hashable_elem(fn_input)
                if hashable_fn_input in bbox.fn_cache:
                    yield bbox.fn_cache[hashable_fn_input]
                else:
                    y = bbox.timeout_decorator(bbox.function)(*fn_input)
                    bbox.fn_cache[hashable_fn_input] = y
                    yield y
        except:
            yield RESERVED_FAILURE

    def process_batch(bbox, batch):
        return list(BlackBoxFunctionFiniteDifference.invoke_function_on_inputs(bbox, batch))

    def invoke_function_on_batched_inputs(bbox, batched_inputs):
        return bbox.pool.map(partial(BlackBoxFunctionFiniteDifference.process_batch, bbox), batched_inputs)

    @staticmethod
    def forward(ctx, bbox, inputs, *input_tensors):

        ctx.save_for_backward(*input_tensors)

        BlackBoxFunctionFiniteDifference.bbox = bbox

        num_inputs = len(inputs)
        assert num_inputs == len(
            bbox.input_mappings), "inputs and input_mappings must have the same length"

        # Get the batch size
        batch_size = inputs[0].batch_size()
        for i in range(1, num_inputs):
            assert batch_size == inputs[i].batch_size(
            ), "all inputs must have the same batch size"

        # Prepare the inputs to the black-box function
        to_compute_inputs, max_indices = [], []
        for (input_i, input_mapping_i) in zip(inputs, bbox.input_mappings):
            max_indices_i, max_elements_i = input_mapping_i.argmax(
                input_i)
            max_indices_i = max_indices_i.reshape((-1, 1))
            to_compute = max_elements_i
            to_compute_inputs.append(to_compute)
            max_indices.append(max_indices_i)

        to_compute_inputs = BlackBoxFunctionFiniteDifference.zip_batched_inputs(
            to_compute_inputs)

        # Get the outputs from the black-box function
        results = BlackBoxFunctionFiniteDifference.invoke_function_on_batched_inputs(
            bbox, to_compute_inputs)

        result_probs = torch.ones(
            (batch_size, 1))
        for i in range(len(max_indices)):
            input_tensor = inputs[i]
            proof = input_tensor.gather(1, max_indices[i])
            result_probs *= proof

        return bbox.output_mapping.vectorize(results, result_probs)

    @staticmethod
    def backward(ctx, om, grad_output, y_pred):
        inputs = ctx.saved_tensors
        js = BlackBoxFunctionFiniteDifference.finite_difference(
            BlackBoxFunctionFiniteDifference.bbox.function, *inputs)
        js = [grad_output.unsqueeze(1).matmul(j).squeeze(1) for j in js]
        return tuple(js)
