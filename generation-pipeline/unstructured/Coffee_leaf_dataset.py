import os
from typing import *
import json
from collections import defaultdict

import torch
from PIL import Image


class CoffeeLeafDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, prefix: str, train: bool):
        super(CoffeeLeafDataset, self).__init__()
        self.img_dir = 'miner_img_xml' if prefix == 'miner' else 'rust_xml_image'
        self.root = root
        self.split = 'train_leaves' if train else 'test_leaves'
        self.metadata = json.load(
            open(os.path.join(root, f"Coffee_leaf/{prefix}_examples.json")))[self.split]
        self.area_dict = json.load(
            open(os.path.join(root, f"Coffee_leaf/{prefix}_areas.json")))
        self.quantiles = self.area_dict['train_quantiles']

    def __getitem__(self, index):
        sample = self.metadata[index]

        sample_path = os.path.join(
            self.root, f"Coffee_leaf/{self.img_dir}/{sample}")
        img_full_path = sample_path + ".jpg"
        img = Image.open(img_full_path).convert("L")
        img_area = self.area_dict[self.split][sample]
        severity = self.get_severity_score(img_area)

        return (img, severity)

    def __len__(self):
        return len(self.metadata)

    def get_severity_score(self, area):
        if area < self.quantiles['Q1']:
            return 0
        elif area < self.quantiles['Q2']:
            return 1
        elif area < self.quantiles['Q3']:
            return 2
        else:
            return 3

    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch)


def get_data(prefix, train):
    severity_scores = [i for i in range(4)]
    data_dir = os.path.abspath(os.path.join(
        os.path.abspath(__file__), "../../data"))
    data = CoffeeLeafDataset(data_dir, prefix, train)

    targets = torch.Tensor([severity for (_, severity) in data])
    sorted = torch.sort(targets)
    idxs = sorted.indices
    values = sorted.values
    ids_of_severity = {}

    for severity in severity_scores:
        t = (values == severity).nonzero(as_tuple=True)[0]
        ids_of_severity[severity] = idxs[t[0]:t[-1]]

    return (data, ids_of_severity)
