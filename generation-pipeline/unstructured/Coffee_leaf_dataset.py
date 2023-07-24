import os
from typing import *
import json
from collections import defaultdict

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional


class BBox:
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def crop_image(self, img):
        area = (self.xmin, self.ymin, self.xmax, self.ymax)
        cropped_img = img.crop(area)
        return cropped_img


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

    def __getitem__(self, index):
        leaf = self.metadata[index]
        print(f"leaf: {leaf}")
        leaf_path = os.path.join(
            self.root, f"Coffee_leaf/{self.img_dir}/{leaf}")
        img_full_path = leaf_path + ".jpg"
        img = Image.open(img_full_path).convert("RGB")

        # get GT severity
        affected_area = self.area_dict[self.split][leaf]
        severity = self.get_severity_score(affected_area)

        # get SAM generated bboxes
        transform = transforms.PILToTensor()
        bboxes = self.sam_bboxes[leaf]
        areas = [0] * len(bboxes)
        images = [None] * len(bboxes)
        for i, bbox in enumerate(bboxes):
            areas[i] = bbox['area']
            box = bbox['bbox']
            crop_area = (box['xmin'], box['ymin'],
                         box['xmax'], box['ymax'])
            cropped_img = img.crop(crop_area)
            resized_img = cropped_img.resize((28, 28))
            images[i] = transform(resized_img)
        return ((images, areas), severity)

    def __len__(self):
        return len(self.metadata)

    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch)


def get_data(prefix: str, train: bool):
    data_dir = os.path.abspath(os.path.join(
        os.path.abspath(__file__), "../../data"))
    data = CoffeeLeafDataset(data_dir, prefix, train)

    ys = [severity for (_, severity) in data]
    sorted_ys = sorted(ys)
    idxs = sorted(range(len(ys)), key=ys.__getitem__)
    ids_of_severity = defaultdict(lambda: [])
    for i in range(len(sorted_ys)):
        ids_of_severity[sorted_ys[i]].append(idxs[i])
    return (data, ids_of_severity)


get_data(prefix='rust', train=True)
