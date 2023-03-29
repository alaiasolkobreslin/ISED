from typing import *

import torch
from torch.distributions.categorical import Categorical

class Sample(object):
    def __init__(self, n_inputs, n_samples, input_mapping, fn):
        self.n_inputs = n_inputs
        self.n_samples = n_samples
        self.input_mapping = input_mapping
        self.fn = fn

    def sample_train(self, inputs, args: Optional[int] = None):

      ground_truth = inputs[self.n_inputs]
      input_distrs = inputs[:self.n_inputs]

      samples = [Categorical(probs=distr).sample((self.n_samples,)) for distr in input_distrs]
      inputs_on = [dict.fromkeys(range(self.input_mapping[i]), 0) for i in range(self.n_inputs)]

      total_on = 0
      for i in range(self.n_samples):
         inputs = [samples[i] for i in range(self.n_inputs)]
         if args is None:
            output = self.fn(inputs)
         else:
            output = self.fn(inputs, args)
         if output[i] == ground_truth:
            total_on += 1
            for j in range(self.n_inputs):
               inputs_on[j][samples[j][i].item()] += 1

      if total_on:
        target_distrs = [torch.tensor([inputs_on[j][i]/total_on for i in range(self.input_mapping[j])]).view(1,-1) for j in range(self.n_inputs)]
      else:
        target_distrs = [torch.zeros((1, self.input_mapping[i])) for i in range(self.n_inputs)]

      return tuple(target_distrs)
    
    def sample_test(self, input_distrs, args: Optional[List] = None):
      batch_size, _ = input_distrs[0].shape
      samples = [torch.t(Categorical(probs=distr).sample((self.n_samples,))) for distr in input_distrs]
      results = torch.zeros(batch_size)
      for i in range(batch_size):
         inputs = [samples[j][i] for j in range(self.n_inputs)]
         if args is None:
            results[i] = torch.mode(self.fn(inputs)).values.item()
         else:
            results[i] = torch.mode(self.fn(inputs, args[0])).values.item()
      return results
