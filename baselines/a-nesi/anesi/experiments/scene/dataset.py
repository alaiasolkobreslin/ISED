import os
from typing import *
import random
from PIL import Image

import torch
import torchvision
import torchvision.transforms.functional
import torch.nn as nn
import torch.nn.functional as F

from openai import OpenAI
import pickle

class SceneNet(nn.Module):
    def __init__(self):
        super(SceneNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.linear1 = nn.Linear(593, 512)
        self.linear2 = nn.Linear(512, 47)
        self.linearp = nn.Linear(81, 81)
    
    def forward(self, x, pred, conf):
        x = F.max_pool2d(F.relu(self.conv1(x)), 3)
        x = F.max_pool2d(F.relu(self.conv2(x)), 3)
        x = F.max_pool2d(F.relu(self.conv3(x)), 3)
        x = x.view(-1, 512)
        pred = torch.cat((pred, conf.unsqueeze(1)), dim=1)
        pred = F.relu(self.linearp(pred))
        x = torch.cat((x, pred), dim=1)
        x = F.relu(self.linear1(x))
        x = F.softmax(self.linear2(x), dim=1)
        return x

yolo = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

yolo_cls = [1, 10, 13, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 56, 57, 58, 59,
            60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]

scenes = ['basement', 'bathroom', 'bedroom', 'dining', 'kitchen', 'lab', 'living', 'lobby', 'office']

lobby_objects = ['coat rack', 'shoe rack', 'umbrella stand', 'key holder', 'shoes', 'mail organizer', 'coats', 'golf bag', 'door', 'welcome sign']

lab_objects = ['microscope', 'fume hood', 'safety goggles', 'machines', 'chemicals', 'cabinets', 'fire extinguisher', 'extension cords', 'machine parts']

bathroom_objects = ['toilet', 'shampoo', 'hairdryer', 'towel', 'shower curtain', 'toothbrush', 'bathtub', 'toilet paper', 'toothpaste', 'soap']

bedroom_objects = ['bed', 'pillows', 'nightstand', 'wardrobe',  'blanket', 'hangers','family photos', 'clothes', 'lamp', 'bag']

living_objects = ['sofa', 'tv', 'fire place', 'gaming console', 'coffee table', 'magazines', 'console table', 'cushion', 
                  'plants', 'piano', 'painting']

kitchen_objects = ['dishwasher', 'refrigerator', 'oven', 'stove', 'kettle', 'sink', 'knives', 'pans', 'cutting board', 'toaster',
                   'dish rack', 'jars', 'coffe maker', 'dish soap']

dining_objects = ['dining table', 'wine glasses', 'placemats', 'wine rack', 'silverware', 'fruit', 'vase',  'dish',  'wine']

office_objects = ['desk', 'computer', 'printer', 'whiteboard', 'stationary', 'paper', 'chair', 'keyboard', 'mouse', 'file cabinet',
                  'rulers']

basement_objects = ['washer', 'storage boxes', 'generator', 'bicycles', 'toolbox', 'lawnmower',  'storage shelves',
                    'board game', 'cleaning supplies', 'luggage',  'sports equipments','camping gear', 'barbeque grill', 'tires']

objects = ["skip", "ball"] + bathroom_objects[:5] + bedroom_objects[:5] + office_objects[:5] + lab_objects[:5] + \
                lobby_objects[:5] + basement_objects[:5] + dining_objects[:5] + kitchen_objects[:5] + living_objects[:5]

objects_long = bathroom_objects + bedroom_objects + office_objects + lab_objects + lobby_objects \
          + basement_objects + dining_objects + kitchen_objects + living_objects + ["skip", "ball"]

class SceneDataset(torch.utils.data.Dataset):
  """
  :param data_root, the root directory of the data folder
  """
  def __init__(
    self,
    data_root: str,
    data_dir: str,
    transform: Optional[Callable] = None,
  ):
    self.transform = transform
    self.labels = scenes
    
    # Get all image paths and their labels
    self.samples = []
    data_dir = os.path.join(data_root, data_dir)
    data_dirs = os.listdir(data_dir)
    for sample_group in data_dirs:
      sample_group_dir = os.path.join(data_dir, sample_group)
      if not os.path.isdir(sample_group_dir) or not sample_group in self.labels:
        continue
      label = self.labels.index(sample_group)
      sample_group_files = os.listdir(sample_group_dir)
      for idx in range(len(sample_group_files)):
        sample_img_path = os.path.join(sample_group_dir, sample_group_files[idx])
        if sample_img_path.endswith('jpg'):
          self.samples.append((sample_img_path, sample_group_files[idx], label))
    
    self.index_map = list(range(len(self.samples)))
    random.shuffle(self.index_map)

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    (img_path, file_name, label) = self.samples[self.index_map[idx]]
    img = Image.open(open(img_path, "rb"))
    img = img.resize((384, 256))
    img = torchvision.transforms.functional.to_tensor(img)
    return (img, file_name, label)
  
  @staticmethod
  def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    names = [item[1] for item in batch]
    labels = torch.stack([torch.tensor(item[2]).long() for item in batch])
    return (imgs, names, labels)

def scene_loader(batch_size):
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../../../finite_diff/data/scene"))
  train_dataset = SceneDataset(data_root, "train")
  test_dataset = SceneDataset(data_root, "test")
  train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=SceneDataset.collate_fn, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=SceneDataset.collate_fn, batch_size=batch_size, shuffle=True)
  return (train_loader, test_loader)

def pad_square(img, fill_color=(0,0,0)):
    x, y = img.size
    size = max(x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
    return new_im.resize((100,100))

def prepare_inputs(img, files, file_dict):
    # Run batched inference on a list of images
    results = []
    for i, file in enumerate(files):
      if file in file_dict: 
        results.append(file_dict[file])
      else:
        raise Exception("No")
    
    box_len, pred, box_list, conf = [], [], [], []
    for n, result in enumerate(results):
      boxes = torch.from_numpy(result.xyxy)
      cls = result.class_id
      confidence = result.confidence
      box_len.append(torch.tensor(len(result)))
      for i, box in enumerate(boxes):
        a, b, c, d = box.int()
        cropped_img = torchvision.transforms.functional.to_pil_image(img[n][:, b:d, a:c])
        square_img = pad_square(cropped_img)
        box_list.append(torchvision.transforms.functional.to_tensor(square_img))
        if cls[i] < 80: 
          pred.append(F.one_hot(torch.tensor(cls[i]), num_classes = 80))
          conf.append(torch.tensor(confidence[i]))
        else: 
          pred.append(torch.zeros(80))
          conf.append(torch.tensor(0))
    box_len = torch.stack(box_len, dim=0)
    pred = torch.stack(pred, dim=0)
    box_list = torch.stack(box_list, dim=0)
    conf = torch.stack(conf, dim=0)
    return (box_len, pred, box_list, conf)

client = OpenAI(
  api_key='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'
)

system_msg = "You are an expert at identifying room types based on the object detected. Give short single responses."
question = "\n What type of room is most likely? Choose among basement, bathroom, bedroom, living room, home lobby, office, lab, kitchen, dining room."
queries = {}

data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../../../finite_diff/scene"))
with open(data_root + '/llm_single.pkl', 'rb') as f: 
  queries = pickle.load(f)

def classify_llm(objects):
  random_scene = ['basement', 'bathroom', 'bedroom', 'dining', 'kitchen', 'lab', 'living', 'lobby', 'office']
  random.shuffle(random_scene)
  answer = call_llm(objects)
  counts = torch.zeros(9)
  for a in answer:
    s = parse_response(a)
    counts[random_scene.index(s)] += 1
  return random_scene[counts.argmax()]

def call_llm(objects):
  r = []
  for o in objects:
    if o == 'skip' or o == 'ball': 
      continue
    prompt = f"There is a {o}."
    if o in queries.keys():
      r.append(queries[o])
      continue
    raise("wrong")
    response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                  {"role": "system", "content": system_msg},
                  {"role": "user", "content": prompt + question}
                ],
                top_p=1e-8
              )
    if response.choices[0].finish_reason == 'stop':
      ans = response.choices[0].message.content.lower()
      print(ans)
      queries[o] = ans
      r.append(ans)
    else: 
      raise Exception("LLM failed to provide an answer") 
  return r

def parse_response(answer):
  random_scene = ['basement', 'bathroom', 'bedroom', 'dining', 'kitchen', 'lab', 'living', 'lobby', 'office']
  random.shuffle(random_scene)
  for s in random_scene:
    if s in answer: return s
  raise Exception("LLM failed to provide an answer") 
