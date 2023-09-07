import os
from typing import *
import json
from collections import defaultdict

import torch
from PIL import Image
from torchvision import transforms


class COFFEE_dataset(torch.utils.data.Dataset):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    def __init__(self, root: str, prefix: str, train: bool):
        super(COFFEE_dataset, self).__init__()
        self.img_dir = 'miner_img_xml' if prefix == 'miner' else 'rust_xml_image'
        self.prefix = prefix
        self.root = root
        self.split = 'train_leaves' if train else 'test_leaves'
        self.metadata = json.load(
            open(os.path.join(root, f"Coffee_leaf/{prefix}_examples.json")))[self.split]
        self.area_dict = json.load(
            open(os.path.join(root, f"Coffee_leaf/{prefix}_areas.json")))
        self.sam_bboxes = json.load(
            open(os.path.join(root, f"Coffee_leaf/{prefix}_generated_bboxes.json")))[self.split]
        self.quantiles = self.area_dict['train_quantiles']

    def get_severity_score(self, area):
        if area < self.quantiles['Q1']:
            return 1
        elif area < self.quantiles['Q2']:
            return 2
        elif area < self.quantiles['Q3']:
            return 3
        elif area < self.quantiles['Q4']:
            return 4
        else:
            return 5

    def get_ids_of_severity(self):
        ys = [self.get_severity_score(
            self.area_dict[self.split][leaf]) for leaf in self.metadata]
        sorted_ys = sorted(ys)
        idxs = sorted(range(len(ys)), key=ys.__getitem__)
        ids_of_severity = defaultdict(lambda: [])
        for i in range(len(sorted_ys)):
            ids_of_severity[sorted_ys[i]].append(idxs[i])
        return ids_of_severity

    def __getitem__(self, index):
        leaf = self.metadata[index]
        leaf_path = os.path.join(
            self.root, f"Coffee_leaf/{self.img_dir}/{leaf}")
        img_full_path = leaf_path + ".jpg"
        img = Image.open(img_full_path).convert("RGB")

        # get GT severity
        affected_area = self.area_dict[self.split][leaf]
        severity = self.get_severity_score(affected_area)

        # get SAM generated bboxes
        bboxes = self.sam_bboxes[leaf]
        areas = []
        images = []
        img_w, img_h = img.size
        for i, bbox in enumerate(bboxes):
            box = bbox['bbox']

            # First, we want to discard sections that are too large
            crop_img_area = (box['xmax'] - box['xmin']) * \
                (box['ymax'] - box['ymin'])
            if crop_img_area > 2500000:
                continue

            areas.append(bbox['area'])
            crop_area = (img_w - box['xmax'], img_h - box['ymax'],
                         img_w - box['xmin'], img_h - box['ymin'])
            cropped_img = img.crop(crop_area)
            resized_img = cropped_img.resize((28, 28))
            images.append(self.transform(resized_img))
        return ((images, areas), severity)

    def __len__(self):
        return len(self.metadata)

    @staticmethod
    def collate_fn(batch):
        max_len = 46
        zero_img = torch.zeros_like(batch[0][0][0])

        def pad_zero(img_seq): return img_seq + \
            [zero_img] * (max_len - len(img_seq))
        img_seqs = torch.stack([torch.stack(pad_zero(img_seq))
                               for (img_seq, _) in batch])
        img_seq_len = torch.stack(
            [torch.tensor(img_seq_len).long() for (_, img_seq_len) in batch])
        return (img_seqs, img_seq_len)


def get_data(prefix: str, train: bool):
    data_dir = os.path.abspath(os.path.join(
        os.path.abspath(__file__), "../../data"))
    data = COFFEE_dataset(data_dir, prefix, train)
    return (data, data.get_ids_of_severity())
