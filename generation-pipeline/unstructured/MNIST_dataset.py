import random
import os
from typing import *

import torch
import torchvision

mnist_img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.1307,), (0.3081,)
    )
])


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        digits: List[int],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        # Contains a MNIST dataset
        self.mnist_dataset = torchvision.datasets.MNIST(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.relevant_digits = list(
            filter(lambda d: d[1] in digits, self.mnist_dataset))
        self.index_map = list(range(len(self.relevant_digits)))
        random.shuffle(self.index_map)
        self.shuffled_digits = [self.relevant_digits[idx]
                                for idx in self.index_map]
        self.targets = torch.tensor([d[1] for d in self.shuffled_digits])

    def __len__(self):
        return len(self.shuffled_digits)

    def __getitem__(self, idx):
        return self.relevant_digits[self.index_map[idx]]

    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch)


def get_data(
    train: bool,
    digits: List[int] = [i for i in range(10)],
):
    data_dir = os.path.abspath(os.path.join(
        os.path.abspath(__file__), "../../data"))
    data = MNISTDataset(
        data_dir,
        digits=digits,
        train=train,
        download=True,
        transform=mnist_img_transform,
    )
    sorted = torch.sort(data.targets)
    idxs = sorted.indices
    values = sorted.values
    ids_of_digit = {}
    for digit in digits:
        t = (values == digit).nonzero(as_tuple=True)[0]
        ids_of_digit[digit] = idxs[t[0]:t[-1]]
    return (data, ids_of_digit)
