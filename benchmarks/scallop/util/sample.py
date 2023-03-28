import torch
from torch.distributions.categorical import Categorical
from functools import reduce

class Sample(object):
    def __init__(self, n_inputs, n_samples, input_mapping, fn):
        self.n_inputs = n_inputs
        self.n_samples = n_samples
        self.input_mapping = input_mapping
        self.fn = fn

    def sample_train(self, inputs):

      ground_truth = inputs[self.n_inputs]
      input_distrs = inputs[:self.n_inputs]

      samples = [Categorical(probs=distr).sample((self.n_samples,)) for distr in input_distrs]
      output = self.fn(samples)
      I_p, I_m = [], []
      for i in range(self.n_samples):
         inputs_probs = [input_distrs[dist][samples[dist][i]] for dist in range(self.n_inputs)]
         output_prob =  reduce(lambda x, y: x*y, inputs_probs)
         if output[i] == ground_truth:
            I_p.append(output_prob)
         else:
            I_m.append(output_prob)
  
      I_p = sum(I_p, start=torch.tensor(0., requires_grad=True)) * 1/self.n_samples
      I_m = sum(I_m,start=torch.tensor(0., requires_grad=True)) * 1/self.n_samples
      return I_p, I_m
    
    def sample_test(self, input_distrs):
      batch_size, _ = input_distrs[0].shape
      samples = [torch.t(Categorical(probs=distr).sample((self.n_samples,))) for distr in input_distrs]
      results = torch.zeros(batch_size)
      for i in range(batch_size):
         inputs = [samples[j][i] for j in range(self.n_inputs)]
         results[i] = torch.mode(self.fn(inputs)).values.item()
      return results
