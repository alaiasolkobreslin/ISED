import os
from typing import *
import json
from collections import defaultdict
import xmltodict

import torch
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.transforms import functional

from Coffee import segment


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
        self.quantiles = self.area_dict['train_quantiles']
        self.dataset = self.generate_dataset()

    def get_bounding_boxes(self, xml_full_path):
        with open(xml_full_path, 'r', encoding='utf-8') as file:
            xml_file = file.read()

        xml_objs = xmltodict.parse(xml_file)['annotation']['object']
        if type(xml_objs) is not list:
            xml_objs = [xml_objs]

        bboxes = []
        for obj in xml_objs:
            bbox = obj['bndbox']
            xmin, xmax = int(bbox['xmin']), int(bbox['xmax'])
            ymin, ymax = int(bbox['ymin']), int(bbox['ymax'])
            bboxes.append(BBox(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax))
        return bboxes

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

    def generate_dataset(self):
        length = self.__len__()
        dataset = [None] * length
        samples_dict = {}
        for i in range(length):
            if i < 148:
                continue
            sample = self.metadata[i]
            print(f"sample: {sample}")
            affected_area = self.area_dict[self.split][sample]
            severity = self.get_severity_score(affected_area)

            sample_path = os.path.join(
                self.root, f"Coffee_leaf/{self.img_dir}/{sample}")
            img_full_path = sample_path + ".jpg"

            img_transform = torchvision.transforms.Compose(
                [transforms.PILToTensor()])

            img = Image.open(img_full_path).convert("RGB")
            img_tensor = img_transform(img)
            img_tensor = functional.rotate(img=img_tensor, angle=180)

            bboxes_dict = segment.segment_anything_bboxes(
                img=img_tensor, img_name=sample)
            samples_dict[sample] = bboxes_dict

        print(json.dumps(samples_dict))

    def __getitem__(self, index):
        return self.dataset[index]

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
    ids_of_leaf = defaultdict(lambda: [])
    for i in range(len(sorted_ys)):
        ids_of_leaf[sorted_ys[i]].append(idxs[i])
    return (data, ids_of_leaf)


get_data(prefix='rust', train=True)
