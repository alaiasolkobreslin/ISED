from typing import Optional
import os
import random
from typing import Optional, Callable
from PIL import Image

import torch
import torchvision

from openai import OpenAI
import json

client = OpenAI(
  api_key=os.environ["OPENAI_API_KEY"]
)

leaves_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class LeavesDataset(torch.utils.data.Dataset):
  """
  :param data_root, the root directory of the data folder
  """
  def __init__(
    self,
    data_root: str,
    data_dir: str,
    n_train: int,
    transform: Optional[Callable] = leaves_img_transform,
  ):
    self.transform = transform
    self.labels = ['Alstonia Scholaris', 'Citrus limon', 'Jatropha curcas', 'Mangifera indica', 'Ocimum basilicum',
                   'Platanus orientalis', 'Pongamia Pinnata', 'Psidium guajava', 'Punica granatum', 'Syzygium cumini', 'Terminalia Arjuna']
    
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
      for idx in random.sample(range(len(sample_group_files)), min(n_train, len(sample_group_files))):
        sample_img_path = os.path.join(sample_group_dir, sample_group_files[idx])
        if sample_img_path.endswith('JPG') or sample_img_path.endswith('png'):
          self.samples.append((sample_img_path, label))
    
    self.index_map = list(range(len(self.samples)))
    random.shuffle(self.index_map)

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    (img_path, label) = self.samples[self.index_map[idx]]
    img = Image.open(open(img_path, "rb"))
    img = self.transform(img)
    return (img, label)

def leaves_loader():
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../../../../data"))

  dataset = LeavesDataset(data_root, "leaf_11", 40)
  num_train = 30*11
  num_test = len(dataset) - num_train
  (train_dataset, test_dataset) = torch.utils.data.random_split(dataset, [num_train, num_test])
  return (train_dataset, test_dataset)

_train_set, _val_set = leaves_loader()
datasets = {
    "train": _train_set,
    "val": _val_set,
    "test": _val_set,
}

def classify_11(margin, shape, texture):
  if margin == 'serrate': return 'Ocimum basilicum'
  elif margin == 'indented': return 'Jatropha curcas'
  elif margin == 'lobed': return 'Platanus orientalis'
  elif margin == 'serrulate': return "Citrus limon"
  elif margin == 'entire':
    if shape == 'ovate': return 'Pongamia Pinnata'
    elif shape == 'lanceolate': return 'Mangifera indica'
    elif shape == 'oblong': return 'Syzygium cumini'
    elif shape == 'obovate': return "Psidium guajava"
    else:
      if texture == 'leathery': return "Alstonia Scholaris"
      elif texture == 'rough': return "Terminalia Arjuna"
      elif texture == 'glossy': return "Citrus limon"
      else: return "Punica granatum"
  else:
    if shape == 'elliptical': return 'Terminalia Arjuna'
    elif shape == 'lanceolate': return "Mangifera indica"
    else: return 'Syzygium cumini'

queries = {}

def classify_llm_11(feature1, feature2, feature3):
  result1 = call_llm(labels, features_1)
  plants1 = parse_response(result1, feature1)
  if len(plants1) == 1: return plants1[0]
  elif len(plants1) == 0: 
    plants1 = labels 
  else:
    results2 = call_llm(plants1, features_2)
    plants2 = parse_response(results2, feature2)
    if len(plants2) == 1: return plants2[0]
    elif len(plants2) == 0: 
      plants2 = plants1 
    results3 = call_llm(plants2, features_3)
    plants3 = parse_response(results3, feature3)
    if len(plants3) == 1: return plants3[0]
    elif len(plants3) == 0: return plants2[random.randrange(len(plants2))]
    else: return plants3[random.randrange(len(plants3))]

def call_llm(plants, features):
  user_list = "* " + "\n* ".join(plants)
  question = "\n\nClassify each into one of: " + ", ".join(features) + "."
  format = "\n\nGive your answer without explanation."
  user_msg = user_list + question
  if user_msg in queries.keys():
    return queries[user_msg]
  response = client.chat.completions.create(
              model="gpt-4-1106-preview",
              messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg + format}
              ],
              top_p=0.00000001
            )
  if response.choices[0].finish_reason == 'stop':
    ans = response.choices[0].message.content
    print(ans[7:-3])
    queries[user_msg] = ans[7:-3]
    return ans[7:-3]
  raise Exception("LLM failed to provide an answer") 

def parse_response(result, target):
  dict = json.loads(result)
  plants = []
  for plant in dict.keys():
    if dict[plant] == target: plants.append(plant)
  return plants

system_msg = "You are an expert in classifying plant species based on the margin, shape, and texture of the leaves. You are designed to output a single JSON."
features_1 = ['entire', 'lobed', 'serrate']
features_2 = ['cordate', 'lanceolate', 'oblong', 'oval', 'ovate', 'palmate']
features_3 = ['glossy', 'papery', 'smooth']
labels = ['Alstonia Scholaris', 'Citrus limon', 'Jatropha curcas', 'Mangifera indica', 'Ocimum basilicum', 
          'Platanus orientalis', 'Pongamia Pinnata', 'Psidium guajava', 'Punica granatum', 'Syzygium cumini', 'Terminalia Arjuna']