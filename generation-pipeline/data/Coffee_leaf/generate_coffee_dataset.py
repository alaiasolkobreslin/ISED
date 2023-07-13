
import os
import json
import numpy as np

import xmltodict


def get_areas(train: bool, prefix: str):
    split = 'train_leaves' if train else 'test_leaves'
    img_dir = 'miner_img_xml' if prefix == 'miner' else 'rust_xml_image'
    dir_path = os.path.dirname(os.path.realpath(__file__))

    metadata = json.load(
        open(os.path.join(dir_path, f"{prefix}_examples.json")))[split]
    area_dict = {}
    areas = []

    for sample in metadata:
        sample_path = os.path.join(
            dir_path, f"{img_dir}/{sample}")
        xml_full_path = sample_path + ".xml"

        xml_data = open(xml_full_path, 'r').read()
        xml_dict = xmltodict.parse(xml_data)
        total_area = 0
        objects = xml_dict['annotation']['object']
        if type(objects) is not list:
            objects = [objects]
        for obj in objects:
            bndbox = obj['bndbox']
            xmin = int(bndbox['xmin'])
            xmax = int(bndbox['xmax'])
            ymin = int(bndbox['ymin'])
            ymax = int(bndbox['ymax'])
            total_area += (xmax - xmin) * (ymax - ymin)

        areas.append(total_area)
        area_dict[sample] = total_area
    return (area_dict, np.array(areas))


def generate_dataset(prefix: str):
    (area_dict_train, areas_train) = get_areas(train=True, prefix=prefix)
    (area_dict_test, _) = get_areas(train=False, prefix=prefix)

    quantile_dict = {}
    quantile_dict['Q1'] = np.quantile(areas_train, .25)
    quantile_dict['Q2'] = np.quantile(areas_train, .5)
    quantile_dict['Q3'] = np.quantile(areas_train, .75)

    json_dict = {
        'train_leaves': area_dict_train,
        'test_leaves': area_dict_test,
        'train_quantiles': quantile_dict,
    }

    print(json.dumps(json_dict))


generate_dataset(prefix='rust')
generate_dataset(prefix='miner')
