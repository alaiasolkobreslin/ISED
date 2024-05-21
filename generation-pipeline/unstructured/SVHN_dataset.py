import random
import os
from typing import *

import torch
import torchvision

svhn_img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.1307,), (0.3081,)
    )
])


class SVHNDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        digits: List[int],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        split = 'train' if train else 'test'
        # Contains a SVHN dataset
        self.svhn_dataset = torchvision.datasets.SVHN(
            root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.index_map = list(range(len(self.svhn_dataset)))
        random.shuffle(self.index_map)

    def __len__(self):
        return len(self.svhn_dataset)

    def __getitem__(self, idx):
        return self.svhn_dataset[self.index_map[idx]]

    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch)


def get_data(train):
    data_dir = os.path.abspath(os.path.join(
        os.path.abspath(__file__), "../../data"))
    data = SVHNDataset(
        data_dir,
        train=train,
        download=True,
        transform=svhn_img_transform,
        digits=[i for i in range(10)]
    )
    labels = torch.from_numpy(data.svhn_dataset.labels)
    sorted = torch.sort(labels)
    idxs = sorted.indices
    values = sorted.values
    ids_of_digit = [None] * 10
    for i in range(10):
        t = (values == i).nonzero(as_tuple=True)[0]
        ids_of_digit[i] = idxs[t[0]:t[-1]]
    return (data.svhn_dataset, ids_of_digit)