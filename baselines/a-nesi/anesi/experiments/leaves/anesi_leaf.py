from typing import List, Optional

import torch

from experiments.leaves.anesi_l import ANeSIBase
from experiments.leaves.dataset import classify_11
from experiments.leaves import LeafState
from experiments.leaves.leafnet import LeafNet
from experiments.leaves.im_leaf import InferenceModelLeaf, IndependentIMLeaf

class LeafModel(ANeSIBase[LeafState]):

    def __init__(self, args):
        self.N = args["N"]
        self.y_encoding = args["y_encoding"]
        self.model = args["model"]
    
        if self.model == "full":
            im = InferenceModelLeaf(self.N,
                                    self.output_dims(self.N, self.y_encoding),
                                    layers=args["layers"],
                                    hidden_size=args["hidden_size"],
                                    prune=args["prune"])
        elif args["model"] == "independent":
            im = IndependentIMLeaf(self.N,
                                    self.output_dims(self.N, self.y_encoding),
                                    layers=args["layers"],
                                    hidden_size=args["hidden_size"])
        super().__init__(im,
                         # Perception network
                         LeafNet(),
                         belief_size=[6, 5, 4],
                         **args)

    def output_dims(self, N: int, y_encoding: str) -> List[int]:
        pass

    def initial_state(self, 
                      P1: torch.Tensor, 
                      P2: torch.Tensor,
                      P3: torch.Tensor,
                      y: Optional[torch.Tensor] = None,
                      w: Optional[torch.Tensor] = None, 
                      generate_w=True) -> LeafState:
        output_dims = self.output_dims(self.N, self.y_encoding)

        w_list = None
        if w is not None:
            w_list = [w[:, i] for i in range(1)]
        y_list = None
        if y is not None:
            y_list = self.preprocess_y(y)
        return LeafState(torch.cat((P1, P2, P3), dim=1), self.N, (y_list, w_list), output_dims, generate_w=generate_w)
    
    def preprocess_y(self, y: torch.Tensor) -> List[torch.Tensor]:
        # output_dims = self.output_dims(self.N, self.y_encoding)
        return [y]
    
    def symbolic_function(self, w: torch.Tensor) -> torch.Tensor:
        margin = w[:,0]
        shape = w[:,1]
        texture = w[:,2]

        return self.op(margin, shape, texture)
    
    def op(self, margin, shape, texture):
        raise NotImplementedError()
    
    def success(self, prediction: List[torch.Tensor], y: torch.Tensor, beam=False) -> torch.Tensor:
        if beam and self.model == 'full':
            prediction = list(map(lambda syi: syi[:, 0], prediction))
        else:
            y = y.unsqueeze(-1)
        prediction = prediction[0]
        return y == prediction

