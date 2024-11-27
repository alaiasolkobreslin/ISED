import math
from typing import List

import torch
from experiments.mnist_op import MNISTModel


class MNISTSumNModel(MNISTModel):

    def output_dims(self, N: int, y_encoding: str) -> List[int]:
        if y_encoding == "base10":
            return [10] * (2 * N)
        elif y_encoding == "base2":
            max_y = 10 ** (2 * N)
            return [1] * math.ceil(math.log2(max_y))
        raise NotImplementedError

    def op(self, *ns) -> torch.Tensor:
        ns = torch.stack(ns, dim=-1)
        return ns.sum(dim=-1)