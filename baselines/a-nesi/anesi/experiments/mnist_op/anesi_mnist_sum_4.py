import math
from typing import List

import torch
from experiments.mnist_op import MNISTModel


class MNISTSum4Model(MNISTModel):

    def output_dims(self, N: int, y_encoding: str) -> List[int]:
        if y_encoding == "base10":
            return [10] * (2 * N)
        elif y_encoding == "base2":
            max_y = 10 ** (2 * N)
            return [1] * math.ceil(math.log2(max_y))
        raise NotImplementedError

    def op(self, n1: torch.Tensor, n2: torch.Tensor, n3: torch.tensor, n4: torch.tensor) -> torch.Tensor:
        return n1 + n2 + n3 + n4