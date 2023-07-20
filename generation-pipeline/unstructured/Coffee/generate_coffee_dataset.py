import os
from typing import *
import json
from collections import defaultdict
import xmltodict

import torch
from PIL import Image
import torchvision
from torchvision import transforms

from . import segment


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

    def __getitem__(self, index):
        sample = self.metadata[index]

        sample_path = os.path.join(
            self.root, f"Coffee_leaf/{self.img_dir}/{sample}")
        img_full_path = sample_path + ".jpg"
        xml_full_path = sample_path + ".xml"

        print(f"image path: {img_full_path}")

        bboxes = self.get_bounding_boxes(xml_full_path)

        img_transform = torchvision.transforms.Compose(
            [transforms.PILToTensor()])

        img = Image.open(img_full_path).convert("RGB")
        img_tensor = img_transform(img)
        total_area = 0
        for bbox in bboxes:
            for areas, _ in segment.segment_anything(
                img=img_tensor,
                xmin=bbox.xmin,
                ymin=bbox.ymin,
                xmax=bbox.xmax,
                ymax=bbox.ymax,
            ):
                total_area += areas
        return (sample, total_area)

    def __len__(self):
        return len(self.metadata)

    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch)


def get_data(prefix: str, train: bool):
    data_dir = os.path.abspath(os.path.join(
        os.path.abspath(__file__), "../../data"))
    data = CoffeeLeafDataset(data_dir, prefix, train)

    area_dict = {}

    for (sample, total_area) in data:
        area_dict[sample] = total_area

    print(json.dumps(area_dict))


get_data(prefix='rust', train=False)
