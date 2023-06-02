from typing import *

import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor as Pool

import permute

DEVICE = torch.device('cpu')

pool = None


class Sample:

    def sample_train(self, inputs):
        pass

    def sample_train_backward(self, data, inputs):
        pass

    def sample_train_backward_non_batch(self, inputs):
        pass

    def sample_train_backward_threaded(self, inputs):
        pass

    def sample_test(self, input_distrs, data):
        pass


class StandardSample(Sample):
    def __init__(self, n_inputs, n_samples, fn, flatten_fns, unflatten_fns, config, n_threads=0):
        self.n_inputs = n_inputs
        self.n_samples = n_samples
        self.fn = fn
        self.flatten_fns = flatten_fns
        self.unflatten_fns = unflatten_fns
        # self.permutations = permute.get_permutations(config)
        self.n_threads = n_threads
        # if n_threads > 0:
        # global pool
        # pool = Pool(self.n_threads)

    def sample_train(self, inputs):
        ground_truth = inputs[self.n_inputs]
        input_distrs = inputs[:self.n_inputs]
        input_sampler = [Categorical(i) for i in input_distrs]
        I_p, I_m = [], []
        for _ in range(self.n_samples):
            idxs = [i.sample() for i in input_sampler]
            idxs_probs = torch.stack([input_distrs[i][idx]
                                     for i, idx in enumerate(idxs)])
            output_prob = torch.prod(idxs_probs, dim=0)
            if self.fn(*idxs) == ground_truth:
                I_p.append(output_prob)
            else:
                I_m.append(output_prob)
        I_p_mean = torch.mean(torch.stack(I_p)) if I_p else torch.tensor(
            0., requires_grad=True)
        I_m_mean = torch.mean(torch.stack(I_m)) if I_m else torch.tensor(
            0., requires_grad=True)
        return I_p_mean, I_m_mean

    def sample_train_backward(self, data, inputs):
        length = len(inputs)
        batch_num = inputs[length-1]
        ground_truth = inputs[length-2]
        input_distrs = inputs[:(length-2)]
        input_sampler = [Categorical(i) for i in input_distrs]

        # ensure gradients are kept
        for distr in input_distrs:
            distr.requires_grad = True
            distr.retain_grad()

        I_p, I_m = [], []
        for _ in range(self.n_samples):
            idxs = [i.sample().item() for i in input_sampler]
            idxs_probs = torch.stack([input_distrs[i][idx]
                                     for i, idx in enumerate(idxs)])
            output_prob = torch.prod(idxs_probs, dim=0)

            inputs_ = []
            last_idx = 0
            for i, (unflatten, n) in enumerate(self.unflatten_fns):
                current_inputs = idxs[last_idx:(n+last_idx)]
                inputs_ += unflatten(current_inputs, data[i], batch_num)
                last_idx += n

            if self.fn(*inputs_) == ground_truth:
                I_p.append(output_prob)
            else:
                I_m.append(output_prob)
        I_p_sum = torch.sum(torch.stack(I_p) if I_p else torch.tensor(
            0., requires_grad=True, device=DEVICE))
        I_m_sum = torch.sum(torch.stack(I_m) if I_m else torch.tensor(
            0., requires_grad=True, device=DEVICE))

        truthy = I_p_sum/(I_p_sum+I_m_sum)
        falsey = I_m_sum/(I_p_sum+I_m_sum)

        I = torch.stack((truthy, falsey))
        I_truth = torch.stack((torch.ones(size=truthy.shape, requires_grad=True, device=DEVICE), torch.zeros(
            size=falsey.shape, requires_grad=True, device=DEVICE)))
        l = F.binary_cross_entropy(I, I_truth)
        l.backward()
        gradients = torch.stack([i.grad for i in input_distrs])
        return gradients

    def sample_train_backward_non_batch(self, inputs):
        ground_truth = inputs[self.n_inputs]
        input_distrs = inputs[:self.n_inputs]
        input_sampler = [Categorical(i) for i in input_distrs]

        # ensure gradients are kept
        for distr in input_distrs:
            distr.requires_grad = True
            distr.retain_grad()
        idxs = [i.sample() for i in input_sampler]
        idxs_probs = torch.stack([input_distrs[i][idx]
                                 for i, idx in enumerate(idxs)])
        output_prob = torch.prod(idxs_probs, dim=0)

        if self.fn(*idxs) == ground_truth:
            l = F.mse_loss(output_prob, torch.ones(
                size=output_prob.shape, requires_grad=True))
        else:
            l = F.mse_loss(output_prob, torch.zeros(
                size=output_prob.shape, requires_grad=True))
        l.backward()
        gradients = torch.stack([i.grad for i in input_distrs])
        return gradients

    def sample_train_backward_threaded(self, inputs):
        assert self.n_threads > 0

        ground_truth = inputs[self.n_inputs]
        input_distrs = inputs[:self.n_inputs]
        input_sampler = [Categorical(i) for i in input_distrs]

        # ensure gradients are kept
        for distr in input_distrs:
            distr.requires_grad = True
            distr.retain_grad()

        I_p, I_m = [], []

        def _proofer():
            idxs = [i.sample() for i in input_sampler]
            idxs_probs = torch.stack([input_distrs[i][idx]
                                     for i, idx in enumerate(idxs)])
            output_prob = torch.prod(idxs_probs, dim=0)
            if self.fn(*idxs) == ground_truth:
                I_p.append(output_prob)
            else:
                I_m.append(output_prob)

        semaphores = [pool.submit(_proofer) for _ in range(self.n_samples)]
        concurrent.futures.wait(
            semaphores, return_when=concurrent.futures.ALL_COMPLETED)

        I_p_sum = torch.sum(torch.stack(I_p) if I_p else torch.tensor(
            0., requires_grad=True, device=DEVICE))
        I_m_sum = torch.sum(torch.stack(I_m) if I_m else torch.tensor(
            0., requires_grad=True, device=DEVICE))

        truthy = I_p_sum/(I_p_sum+I_m_sum)
        falsey = I_m_sum/(I_p_sum+I_m_sum)

        I = torch.stack((truthy, falsey))
        I_truth = torch.stack((torch.ones(size=truthy.shape, requires_grad=True, device=DEVICE), torch.zeros(
            size=falsey.shape, requires_grad=True, device=DEVICE)))
        l = F.mse_loss(I, I_truth)
        l.backward()
        gradients = torch.stack([i.grad for i in input_distrs])
        return gradients

    def sample_test(self, input_distrs, data):
        flattened = [flatten(input_distrs[i])
                     for i, flatten in enumerate(self.flatten_fns)]

        batch_size, _ = flattened[0][0].shape
        results = [None] * batch_size
        for i in range(batch_size):
            inputs = []
            for j, (unflatten, _) in enumerate(self.unflatten_fns):
                current_inputs = [torch.argmax(distr[i]).item()
                                  for distr in flattened[j]]
                inputs += unflatten(current_inputs, data[j], i)
            results[i] = self.fn(*inputs)
        return results


class SamplePaddedInput(Sample):
    def __init__(self, n_inputs, n_samples, fn, flatten_fns, unflatten_fns, config, n_threads=0):
        self.n_inputs = n_inputs
        self.n_samples = n_samples
        self.fn = fn
        self.flatten_fns = flatten_fns
        self.unflatten_fns = unflatten_fns
        self.n_threads = n_threads
        # if n_threads > 0:
        # global pool
        # pool = Pool(self.n_threads)

    def sample_train(self, inputs):
        pass

    def sample_train_backward(self, data, inputs):
        length = len(inputs)
        batch_num = inputs[length-1]
        ground_truth = inputs[length-2]
        input_distrs = inputs[:(length-2)]
        input_sampler = [Categorical(i) for i in input_distrs]

        # ensure gradients are kept
        for distr in input_distrs:
            distr.requires_grad = True
            distr.retain_grad()

        I_p, I_m = [], []
        for _ in range(self.n_samples):
            idxs = [i.sample() for i in input_sampler]
            # TODO: this is wrong. We are using all idxs instead of the
            # relevant ones
            # idxs_probs = torch.stack([input_distrs[i][idx]
            #                          for i, idx in enumerate(idxs)])
            # output_prob = torch.prod(idxs_probs, dim=0)

            inputs_ = []
            last_idx = 0
            relevant_distrs = []
            relevant_distrs_for_grad = []

            for i, (unflatten, n) in enumerate(self.unflatten_fns):

                length = data[i][1][batch_num].item()
                relevant_idxs = idxs[last_idx:(length + last_idx)]
                relevant_distrs += [input_distrs[i + last_idx][idx]
                                    for i, idx in enumerate(relevant_idxs)]
                relevant_distrs_for_grad += [input_distrs[i + last_idx]
                                             for i, _ in enumerate(relevant_idxs)]

                current_inputs = idxs[last_idx:(n+last_idx)]
                inputs_ += unflatten(current_inputs, data[i], batch_num)
                last_idx += n

            output_prob = torch.prod(torch.stack(relevant_distrs), dim=0)

            if self.fn(*inputs_) == ground_truth:
                I_p.append(output_prob)
            else:
                I_m.append(output_prob)
        I_p_sum = torch.sum(torch.stack(I_p) if I_p else torch.tensor(
            0., requires_grad=True, device=DEVICE))
        I_m_sum = torch.sum(torch.stack(I_m) if I_m else torch.tensor(
            0., requires_grad=True, device=DEVICE))

        truthy = I_p_sum/(I_p_sum+I_m_sum)
        falsey = I_m_sum/(I_p_sum+I_m_sum)

        I = torch.stack((truthy, falsey))
        I_truth = torch.stack((torch.ones(size=truthy.shape, requires_grad=True, device=DEVICE), torch.zeros(
            size=falsey.shape, requires_grad=True, device=DEVICE)))
        l = F.binary_cross_entropy(I, I_truth)
        l.backward()

        # for distr in relevant_distrs_for_grad:
        #     distr.requires_grad = True
        #     distr.retain_grad()

        gradients = torch.stack([i.grad for i in relevant_distrs_for_grad])
        return gradients

    def sample_train_backward_non_batch(self, inputs):
        pass

    def sample_train_backward_threaded(self, inputs):
        pass

    def sample_test(self, input_distrs, data):
        flattened = [flatten(input_distrs[i])
                     for i, flatten in enumerate(self.flatten_fns)]

        batch_size, _ = flattened[0][0].shape
        results = [None] * batch_size
        for i in range(batch_size):
            inputs = []
            for j, (unflatten, _) in enumerate(self.unflatten_fns):
                current_inputs = [torch.argmax(distr[i])
                                  for distr in flattened[j]]
                inputs += unflatten(current_inputs, data[j], i)
            results[i] = self.fn(*inputs)
        return results
