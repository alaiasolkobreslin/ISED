from json import dumps

import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model

from torch.utils.data import Dataset as TorchDataset
from deepproblog.dataset import Dataset
from deepproblog.query import Query
from typing import *
from pathlib import Path
import torchvision
import random
import itertools
import json
from problog.logic import Term, list2term, Constant
import os
from PIL import Image
from torch import nn

from argparse import ArgumentParser

import torch.nn as nn
import torch.nn.functional as F


leaves_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class Leaf_Net_Margin(nn.Module):
  def __init__(self):
    super(Leaf_Net_Margin, self).__init__()

    # features for classification
    self.f1 = ['entire', 'indented', 'lobed', 'serrate', 'serrulate', 'undulate']
    self.dim = 2304
  
    # CNN
    self.cnn = nn.Sequential(
      nn.Conv2d(3, 32, 10, 1),
      nn.ReLU(),
      nn.MaxPool2d(3),
      nn.Conv2d(32, 64, 5, 1),
      nn.ReLU(),
      nn.MaxPool2d(3),
      nn.Conv2d(64, 128, 3, 1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(128, 128, 3, 1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Flatten(),
    )
    
    # Fully connected for 'f1'
    self.f1_fc = nn.Sequential(
      nn.Linear(self.dim, self.dim),
      nn.ReLU(),
      nn.Linear(self.dim, len(self.f1)),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    x = self.cnn(x)
    has_f1 = self.f1_fc(x)
    return has_f1

class Leaf_Net_Shape(nn.Module):
  def __init__(self):
    super(Leaf_Net_Shape, self).__init__()

    # features for classification
    self.f2 = ['elliptical', 'lanceolate', 'oblong', 'obovate', 'ovate']
    self.dim = 2304
  
    # CNN
    self.cnn = nn.Sequential(
      nn.Conv2d(3, 32, 10, 1),
      nn.ReLU(),
      nn.MaxPool2d(3),
      nn.Conv2d(32, 64, 5, 1),
      nn.ReLU(),
      nn.MaxPool2d(3),
      nn.Conv2d(64, 128, 3, 1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(128, 128, 3, 1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Flatten(),
    )
    
    # Fully connected for 'f2'
    self.f2_fc = nn.Sequential(
      nn.Linear(self.dim, self.dim),
      nn.ReLU(),
      nn.Linear(self.dim, len(self.f2)),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    x = self.cnn(x)
    has_f2 = self.f2_fc(x)
    return has_f2

class Leaf_Net_Texture(nn.Module):
  def __init__(self):
    super(Leaf_Net_Texture, self).__init__()

    # features for classification
    self.f3 = ['glossy', 'leathery', 'smooth', 'rough']
    self.dim = 2304
  
    # CNN
    self.cnn = nn.Sequential(
      nn.Conv2d(3, 32, 10, 1),
      nn.ReLU(),
      nn.MaxPool2d(3),
      nn.Conv2d(32, 64, 5, 1),
      nn.ReLU(),
      nn.MaxPool2d(3),
      nn.Conv2d(64, 128, 3, 1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(128, 128, 3, 1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Flatten(),
    )
    
    # Fully connected for 'f3'
    self.f3_fc = nn.Sequential(
      nn.Linear(self.dim, self.dim),
      nn.ReLU(),
      nn.Linear(self.dim, len(self.f3)),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    x = self.cnn(x)
    has_f3 = self.f1_3c(x)
    return has_f3


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
    data_root = os.path.dirname(os.path.realpath(__name__))
    data_dir = os.path.join(data_root + "/../../data", data_dir)
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
  
  @staticmethod
  def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.stack([torch.tensor(item[1]).long() for item in batch])
    return (imgs, labels)
  
dir_path = os.path.dirname(os.path.realpath(__name__))
data_root = os.path.join(dir_path, '../../data')
leaf_dataset = LeavesDataset(data_root, 'leaf_11', 40)
num_train = 330
num_test = len(leaf_dataset) - num_train
(train_dataset, test_dataset) = torch.utils.data.random_split(leaf_dataset, [num_train, num_test])

datasets = {
    "train": train_dataset,
    "test": test_dataset,
}

class LeafOperator(Dataset, TorchDataset):
    def __getitem__(self, index: int) -> Tuple[list, list, int]:
        l1, l2 = self.data[index]
        label = self._get_label(index)
        l1 = [self.dataset[x][0] for x in l1]
        l2 = [self.dataset[x][0] for x in l2]
        return l1, l2, label

    def __init__(
        self,
        dataset_name: str,
        function_name: str,
        operator: Callable[[List[int]], int],
        size=1,
        arity=2,
        seed=None,
    ):
        """Generic dataset for operator(img, img) style datasets.

        :param dataset_name: Dataset to use (train, val, test)
        :param function_name: Name of Problog function to query.
        :param operator: Operator to generate correct examples
        :param size: Size of numbers (number of digits)
        :param arity: Number of arguments for the operator
        :param seed: Seed for RNG
        """
        super(LeafOperator, self).__init__()
        assert size >= 1
        assert arity >= 1
        self.dataset_name = dataset_name
        self.dataset = datasets[self.dataset_name]
        self.function_name = function_name
        self.operator = operator
        self.size = size
        self.arity = arity
        self.seed = seed
        leaf_indices = list(range(len(self.dataset)))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(leaf_indices)
        dataset_iter = iter(leaf_indices)
        # Build list of examples (mnist indices)
        self.data = []
        try:
            while dataset_iter:
                self.data.append(
                    [
                        [next(dataset_iter) for _ in range(self.size)]
                        for _ in range(self.arity)
                    ]
                )
        except StopIteration:
            pass

    def to_file_repr(self, i):
        """Old file represenation dump. Not a very clear format as multi-digit arguments are not separated"""
        return f"{tuple(itertools.chain(*self.data[i]))}\t{self._get_label(i)}"

    def to_json(self):
        """
        Convert to JSON, for easy comparisons with other systems.

        Format is [EXAMPLE, ...]
        EXAMPLE :- [ARGS, expected_result]
        ARGS :- [MULTI_DIGIT_NUMBER, ...]
        MULTI_DIGIT_NUMBER :- [mnist_img_id, ...]
        """
        data = [(self.data[i], self._get_label(i)) for i in range(len(self))]
        return json.dumps(data)

    def to_query(self, i: int) -> Query:
        """Generate queries"""
        leaf_indices = self.data[i]
        expected_result = self._get_label(i)

        # Build substitution dictionary for the arguments
        subs = dict()
        var_names = []
        for i in range(self.arity):
            inner_vars = []
            for j in range(self.size):
                t = Term(f"p{i}_{j}")
                subs[t] = Term(
                    "tensor",
                    Term(
                        self.dataset_name,
                        Constant(leaf_indices[i][j]),
                    ),
                )
                inner_vars.append(t)
            var_names.append(inner_vars)

        # Build query
        if self.size == 1:
            return Query(
                Term(
                    self.function_name,
                    *(e[0] for e in var_names),
                    Constant(expected_result),
                ),
                subs,
            )
        else:
            return Query(
                Term(
                    self.function_name,
                    *(list2term(e) for e in var_names),
                    Constant(expected_result),
                ),
                subs,
            )

    def _get_label(self, i: int):
        leaf_indices = self.data[i]
        # Figure out what the ground truth is, first map each parameter to the value:
        ground_truth = [
            self.dataset[i[0]][1] for i in leaf_indices
        ]
        # Then compute the expected value:
        expected_result = self.operator(ground_truth)
        return expected_result

    def __len__(self):
        return len(self.data)
    
class Leaf_Images(object):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, item):
        return datasets[self.subset][int(item[0])][0]

Leaf_train = Leaf_Images("train")
Leaf_test = Leaf_Images("test")


def leaf_11(n: int, dataset: str, seed=None):
    return LeafOperator(
        dataset_name=dataset,
        function_name="main",
        operator=lambda x: x[0], #define solution here
        size=n,
        arity=1,
        seed=seed,
    )


parser = ArgumentParser("leaf_11")
parser.add_argument("--seed", type=int, default=1234)
args = parser.parse_args()

method = "exact"
N = 1

name = "leaf_11{}_{}_{}".format(method, N, args.seed)

train_set = leaf_11(N, "train")
test_set = leaf_11(N, "test")

# train_set = train_set.subset(0, 10000)
# test_set = test_set.subset(0, 1000)

network_margin = Leaf_Net_Margin()
network_shape = Leaf_Net_Shape()
network_texture = Leaf_Net_Texture()

net_margin = Network(network_margin, "leaf_net_margin", batching=True)
net_shape = Network(network_margin, "leaf_net_shape", batching=True)
net_texture = Network(network_margin, "leaf_net_texture", batching=True)
net_margin.optimizer = torch.optim.Adam(network_margin.parameters(), lr=1e-3)
net_shape.optimizer = torch.optim.Adam(network_shape.parameters(), lr=1e-3)
net_texture.optimizer = torch.optim.Adam(network_texture.parameters(), lr=1e-3)

model = Model("leaf/models/leaf_11.pl", [net_margin, net_shape, net_texture])
if method == "exact":
    model.set_engine(ExactEngine(model), cache=False)
elif method == "geometric_mean":
    model.set_engine(
        ApproximateEngine(model, 1, ApproximateEngine.geometric_mean, exploration=False)
    )

model.add_tensor_source("train", Leaf_train)
model.add_tensor_source("test", Leaf_test)

loader = DataLoader(train_set, 2, False, seed=args.seed)
train = train_model(model, loader, 30, log_iter=100, profile=0)
model.save_state("snapshot/" + name + ".pth")
train.logger.comment(dumps(model.get_hyperparameters()))
train.logger.comment(
    "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy())
)
train.logger.write_to_file("log/" + name)