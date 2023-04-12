from typing import *

from typing import *

import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor as Pool


pool = None


class Sample(object):
    def __init__(self, n_inputs, n_samples, fn, structured_datasets, n_threads=0):
        self.n_inputs = n_inputs
        self.n_samples = n_samples
        self.fn = fn
        self.structured_datasets = structured_datasets
        self.n_threads = n_threads
        if n_threads > 0:
            global pool
            pool = Pool(self.n_threads)

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

    def sample_train_backward(self, inputs):
        ground_truth = inputs[self.n_inputs]
        input_distrs = inputs[:self.n_inputs]
        input_sampler = [Categorical(i) for i in input_distrs]

        # ensure gradients are kept
        for distr in input_distrs:
            distr.requires_grad = True
            distr.retain_grad()

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

        I = torch.stack((I_p_mean, I_m_mean))
        I_truth = torch.stack((torch.ones(size=I_p_mean.shape, requires_grad=True), torch.zeros(
            size=I_m_mean.shape, requires_grad=True)))
        l = F.mse_loss(I, I_truth)
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

        I_p_mean = torch.mean(torch.stack(I_p)) if I_p else torch.tensor(
            0., requires_grad=True)
        I_m_mean = torch.mean(torch.stack(I_m)) if I_m else torch.tensor(
            0., requires_grad=True)

        I = torch.stack((I_p_mean, I_m_mean))
        I_truth = torch.stack((torch.ones(size=I_p_mean.shape, requires_grad=True), torch.zeros(
            size=I_m_mean.shape, requires_grad=True)))
        l = F.mse_loss(I, I_truth)
        l.backward()
        gradients = torch.stack([i.grad for i in input_distrs])
        return gradients

    def sample_test(self, input_distrs):
        flattened = [structured.flatten(input_distrs[i]) for i, structured in enumerate(self.structured_datasets)]

        batch_size, _ = flattened[0][0].shape
        results = [None] * batch_size
        for i in range(batch_size):
            inputs = []
            for j, structured in enumerate(self.structured_datasets):
                current_inputs = [torch.argmax(distr[j]) for distr in flattened[j]]
                inputs += structured.unflatten(current_inputs)
            results[i] = self.fn(*inputs)
        return results
