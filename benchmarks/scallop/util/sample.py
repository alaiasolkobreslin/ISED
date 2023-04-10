import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F


class Sample(object):
    def __init__(self, n_inputs, n_samples, input_mapping, fn):
        self.n_inputs = n_inputs
        self.n_samples = n_samples
        self.input_mapping = input_mapping
        self.fn = fn

    def sample_train(self, inputs):
      ground_truth = inputs[self.n_inputs]
      input_distrs = inputs[:self.n_inputs]
      input_sampler = [Categorical(i) for i in input_distrs]
      I_p, I_m = [], []
      for _ in range(self.n_samples):
         idxs = [i.sample() for i in input_sampler]
         idxs_probs = torch.stack([input_distrs[i][idx] for i, idx in enumerate(idxs)])
         output_prob = torch.prod(idxs_probs, dim=0)
         if self.fn(idxs) == ground_truth:
            I_p.append(output_prob)
         else:
            I_m.append(output_prob)
      I_p_mean = torch.mean(torch.stack(I_p)) if I_p else torch.tensor(0., requires_grad=True)
      I_m_mean = torch.mean(torch.stack(I_m)) if I_m else torch.tensor(0., requires_grad=True)
      return I_p_mean, I_m_mean
    
    def sample_test(self, input_distrs):
      batch_size, _ = input_distrs[0].shape
      samples = [torch.t(Categorical(probs=distr).sample((self.n_samples,))) for distr in input_distrs]
      results = torch.zeros(batch_size)
      for i in range(batch_size):
         inputs = [samples[j][i] for j in range(self.n_inputs)]
         results[i] = torch.mode(self.fn(inputs)).values.item()
      return results
    
    def sample_train_backward(self, inputs):
      ground_truth = inputs[self.n_inputs]
      input_distrs = inputs[:self.n_inputs]
      input_sampler = [Categorical(i) for i in input_distrs]
      
      #ensure gradients are kept
      for distr in input_distrs:
         distr.requires_grad = True
         distr.retain_grad()

      I_p, I_m = [], []
      for _ in range(self.n_samples):
         idxs = [i.sample() for i in input_sampler]
         idxs_probs = torch.stack([input_distrs[i][idx] for i, idx in enumerate(idxs)])
         output_prob = torch.prod(idxs_probs, dim=0)
         if self.fn(idxs) == ground_truth:
            I_p.append(output_prob)
         else:
            I_m.append(output_prob)
      I_p_mean = torch.mean(torch.stack(I_p)) if I_p else torch.tensor(0., requires_grad=True)
      I_m_mean = torch.mean(torch.stack(I_m)) if I_m else torch.tensor(0., requires_grad=True)

      I = torch.stack((I_p_mean, I_m_mean))
      I_truth = torch.stack((torch.ones(size=I_p_mean.shape, requires_grad=True), torch.zeros(size=I_m_mean.shape, requires_grad=True)))
      l = F.mse_loss(I, I_truth)
      l.backward()
      gradients = torch.stack([i.grad for i in input_distrs])
      return gradients
    
    def sample_test(self, input_distrs):
      batch_size, _ = input_distrs[0].shape
      samples = [torch.t(Categorical(probs=distr).sample((self.n_samples,))) for distr in input_distrs]
      results = torch.zeros(batch_size)
      for i in range(batch_size):
         inputs = [samples[j][i] for j in range(self.n_inputs)]
         results[i] = torch.mode(self.fn(inputs)).values.item()
      return results

