import math
from typing import List

import torch
from experiments.mnist_op import MNISTModel


class MNISTHowMany3Or4Model(MNISTModel):

    def output_dims(self, N: int, y_encoding: str) -> List[int]:
        if y_encoding == "base10":
            return [10] * (2 * N)
        elif y_encoding == "base2":
            max_y = 10 ** (2 * N)
            return [1] * math.ceil(math.log2(max_y))
        raise NotImplementedError

    def op(self, n1, n2, n3, n4, n5, n6, n7, n8) -> torch.Tensor:
        all_tensors = torch.stack([n1, n2, n3, n4, n5, n6, n7, n8])
        is_three = all_tensors == 3
        is_four = all_tensors == 4
        return torch.sum(is_three | is_four, dim=0)