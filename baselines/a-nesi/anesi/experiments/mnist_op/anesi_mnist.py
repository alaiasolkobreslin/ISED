from typing import List, Optional

import torch

from anesi import ANeSIBase
from experiments.MNISTNet import MNIST_Net
from experiments.mnist_op import MNISTState
from experiments.mnist_op.im_mnist_op import InferenceModelMnist, IndependentIMMnist


class MNISTModel(ANeSIBase[MNISTState]):

    def __init__(self, args):
        self.N = args["N"]
        self.arity = args["arity"]
        self.y_encoding = args["y_encoding"]
        self.model = args["model"]

        if self.model == "full":
            im = InferenceModelMnist(self.N,
                                     self.arity,
                                     self.output_dims(self.N, self.y_encoding),
                                     layers=args["layers"],
                                     hidden_size=args["hidden_size"],
                                     prune=args["prune"])
        elif args["model"] == "independent":
            im = IndependentIMMnist(self.N,
                                    self.output_dims(self.N, self.y_encoding),
                                    layers=args["layers"],
                                    hidden_size=args["hidden_size"])
        super().__init__(im,
                         # Perception network
                         MNIST_Net(),
                         belief_size=[10] * self.arity * self.N,
                         **args)

    def output_dims(self, N: int, y_encoding: str) -> List[int]:
        pass

    def initial_state(self,
                      P: torch.Tensor,
                      y: Optional[torch.Tensor] = None,
                      w: Optional[torch.Tensor] = None,
                      generate_w=True) -> MNISTState:
        output_dims = self.output_dims(self.N, self.y_encoding)
        w_list = None
        if w is not None:
            w_list = [w[:, i] for i in range(self.N * self.arity)]
        y_list = None
        if y is not None:
            y_list = self.preprocess_y(y)
        return MNISTState(P, self.N, (y_list, w_list), output_dims, generate_w=generate_w)

    def preprocess_y(self, y: torch.Tensor) -> List[torch.Tensor]:
        output_dims = self.output_dims(self.N, self.y_encoding)
        base = 10 if self.y_encoding == "base10" else 2
        y_list = [(torch.div(y, (base ** (len(output_dims) - 1 - i)), rounding_mode='floor') % base).long()
                  for i in range(len(output_dims))]
        return y_list

    def symbolic_function(self, w: torch.Tensor) -> torch.Tensor:
        """
        w: (batch_size, 2*n)
        """
        stacks = []
        for i in range(self.arity):
            stacks.append(torch.stack([10 ** (self.N - i - 1) * w[..., self.N * j + i] for j in range(self.N)], -1))

        ns = [stack.sum(-1) for stack in stacks]

        return self.op(*ns)

    def op(self, n1: torch.Tensor, n2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def success(self, prediction: List[torch.Tensor], y: torch.Tensor, beam=False) -> torch.Tensor:
        if beam and self.model == "full":
            prediction = list(map(lambda syi: syi[:, 0], prediction))
        else:
            y = y.unsqueeze(-1)
        len_y_enc = len(self.output_dims(self.N, self.y_encoding))
        base = 10 if self.y_encoding == "base10" else 2
        stack = torch.stack([base ** (len_y_enc - 1 - i) * prediction[i] for i in range(len_y_enc)], -1)
        return stack.sum(-1) == y
