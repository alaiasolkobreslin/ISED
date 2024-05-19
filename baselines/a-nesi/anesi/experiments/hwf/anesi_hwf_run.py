import math
from typing import Optional, List

import torch

EPS = 1E-6

from experiments.hwf.dataset import hwf_eval
from experiments.hwf import _HWFState
from experiments.hwf.anesi_hwf import ANeSIBase
from experiments.hwf.dataset import SymbolNet
from experiments.hwf.im_hwf import InferenceModelHWF, IndependentIMHWF

class _HWFModel(ANeSIBase[_HWFState]):

    def __init__(self, args):
        self.N = args["N"]
        self.y_encoding = args["y_encoding"]
        self.model = args["model"]

        if self.model == "full":
            im = InferenceModelHWF(self.N,
                                    self.output_dims(self.N, self.y_encoding),
                                    layers=args["layers"],
                                    hidden_size=args["hidden_size"],
                                    prune=args["prune"])
        elif args["model"] == "independent":
            im = IndependentIMHWF(self.N,
                                  self.output_dims(self.N, self.y_encoding),
                                  layers=args["layers"],
                                  hidden_size=args["hidden_size"])
        super().__init__(im,
                         # Perception network
                         SymbolNet(),
                         belief_size=[14] * 7 * self.N,
                         **args)

    def output_dims(self, N: int, y_encoding: str) -> List[int]:
        pass

    def initial_state(self,
                      P: torch.Tensor,
                      y: Optional[torch.Tensor] = None,
                      w: Optional[torch.Tensor] = None,
                      generate_w=True) -> _HWFState:
        output_dims = self.output_dims(self.N, self.y_encoding)
        
        w_list = None
        if w is not None:
            w_list = [w[:, i] for i in range(self.N * 7)]
        y_list = None
        if y is not None:
            y_list = self.preprocess_y(y)
        return _HWFState(P, self.N, (y_list, w_list), output_dims, generate_w=generate_w)

    def preprocess_y(self, y: torch.Tensor) -> List[torch.Tensor]:
        ys = torch.zeros(23, len(y))
        for i, yi in enumerate(y):
            if yi == 10000: continue
            if yi >= 0: 
                ys[0,i] = 1
            num = '{:023.18f}'.format(abs(yi)).split('.')
            for ni, numi in enumerate(num[0]+num[1]):
                ys[ni+1,i] = int(numi)
    # ys = ys.tolist()
        return ys.long()

    def symbolic_function(self, w: torch.Tensor, eqn_len) -> torch.Tensor:
        """
        w: (batch_size, 2*n)
        """
        return self.op(w, eqn_len)

    def op(self, n1: torch.Tensor, n2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def success(self, prediction: List[torch.Tensor], y: torch.Tensor, beam=False) -> torch.Tensor:
        if beam and self.model == "full":
            prediction = list(map(lambda syi: syi[:, 0], prediction))
        else:
            y = y.unsqueeze(-1)
        stack = torch.stack(prediction, dim=0).t()
        mult = torch.tensor((1000, 100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 
                1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18))
        new_stack = []
        for s in stack:
            if s.sum() == 0: 
                new_stack.append(torch.tensor(10000))
                continue
            if s[0] == 0: sign = -1
            else: sign = 1
            new_stack.append((s[1:] * mult).sum() * sign)
        stack = torch.stack(new_stack)
        return abs(stack-y) < 0.01

class HWFState(_HWFState):

    def len_y_encoding(self):
        return 7
    
class HWFModel(_HWFModel):

    def initial_state(self,
                      P: torch.Tensor,
                      y: Optional[torch.Tensor] = None,
                      w: Optional[torch.Tensor] = None,
                      generate_w=True) -> HWFState:
        
        _initialstate = super().initial_state(P, y, w, generate_w)
        return HWFState(_initialstate.pw, _initialstate.N, _initialstate.constraint, _initialstate.y_dims, generate_w=generate_w)
    
    def op(self, ns, eqn_len) -> torch.Tensor:
        result = []
        ns = ns.reshape(-1, len(eqn_len), 7)
        for n in ns:
            for i in range(len(eqn_len)):
                try:
                    inputs = n[i, :eqn_len[i]]
                    inputs = [symbols[int(j)] for j in inputs]
                    y = hwf_eval(inputs, len(inputs))
                    result.append(torch.tensor(y))
                except:
                    result.append(torch.tensor(10000))
        return torch.stack(result, dim=0)

    def output_dims(self, N:int, y_encoding:str):
        return [1] + [10] * 22

symbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', ]
 