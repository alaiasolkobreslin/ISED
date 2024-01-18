import os
from typing import *
import json
from collections import defaultdict

import torch
import torchvision
from PIL import Image


class HWFDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, prefix: str, split: str):
        super(HWFDataset, self).__init__()
        self.root = root
        self.split = split
        self.metadata = json.load(
            open(os.path.join(root, f"HWF/{prefix}_{split}.json")))
        self.img_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (1,))
        ])

    def __getitem__(self, index):
        sample = self.metadata[index]

        # Input is a sequence of images
        img_seq = []
        for img_path in sample["img_paths"]:
            img_full_path = os.path.join(
                self.root, "HWF/Handwritten_Math_Symbols", img_path)
            img = Image.open(img_full_path).convert("L")
            img = self.img_transform(img)
            img_seq.append(img)
        img_seq_len = len(img_seq)

        # Output is the "res" in the sample of metadata
        expr = sample["expr"]
        res = sample["res"]

        # Return (input, output) pair
        return (img_seq, expr, res)

    def __len__(self):
        return len(self.metadata)

    @staticmethod
    def collate_fn(batch):
        max_len = 7
        zero_img = torch.zeros_like(batch[0][0][0])

        def pad_zero(img_seq): return img_seq + \
            [zero_img] * (max_len - len(img_seq))
        img_seqs = torch.stack([torch.stack(pad_zero(img_seq))
                               for (img_seq, _) in batch])
        img_seq_len = torch.stack(
            [torch.tensor(img_seq_len).long() for (_, img_seq_len) in batch])
        return (img_seqs, img_seq_len)


def get_data(train):
    prefix = "expr"
    split = "train" if train else "test"
    data_dir = os.path.abspath(os.path.join(
        os.path.abspath(__file__), "../../data"))
    data = HWFDataset(data_dir, prefix, split)
    ys = [m['expr'] for m in data.metadata]
    sorted_ys = sorted(ys)
    idxs = sorted(range(len(ys)), key=ys.__getitem__)
    ids_of_expr = defaultdict(lambda: [])
    for i in range(len(sorted_ys)):
        ids_of_expr[sorted_ys[i]].append(idxs[i])
    return (data, ids_of_expr)
