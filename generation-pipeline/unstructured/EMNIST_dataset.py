import random
import os
from typing import *

import torch
import torchvision

emnist_img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.1307,), (0.3081,)
    )
])


class EMNISTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        # Contains an EMNIST dataset
        self.emnist_dataset = torchvision.datasets.EMNIST(
            root,
            split='balanced',
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.index_map = list(range(len(self.emnist_dataset)))
        random.shuffle(self.index_map)

    def __len__(self):
        return len(self.emnist_dataset)

    def __getitem__(self, idx):
        return self.emnist_dataset[self.index_map[idx]]

    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch)


def get_data(train):
    data_dir = os.path.abspath(os.path.join(
        os.path.abspath(__file__), "../../data"))
    data = EMNISTDataset(
        data_dir,
        train=train,
        download=True,
        transform=emnist_img_transform,
    )
    sorted = torch.sort(data.emnist_dataset.targets)
    idxs = sorted.indices
    values = sorted.values
    ids_of_character = [None] * 47
    for i in range(47):
        t = (values == i).nonzero(as_tuple=True)[0]
        ids_of_character[i] = idxs[t[0]:t[-1]]
    return (data.emnist_dataset, ids_of_character)
