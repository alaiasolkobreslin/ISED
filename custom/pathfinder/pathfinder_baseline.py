from typing import Optional, Callable, Tuple
import os
import random
import itertools

import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from PIL import Image

from argparse import ArgumentParser
from tqdm import tqdm

pathfinder_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class PathFinder32Dataset(torch.utils.data.Dataset):
  """
  :param data_root, the root directory of the data folder
  :param data_dir, the directory to the pathfinder dataset under the root folder
  :param difficulty, can be picked from "easy", "normal", "hard", and "all"
  """
  def __init__(
    self,
    data_root: str,
    data_dir: str = "pathfinder32",
    difficulty: str = "all",
    transform: Optional[Callable] = pathfinder_img_transform,
  ):
    # Store
    self.transform = transform

    # Get subdirectories
    if difficulty == "all":
      sub_dirs = ["curv_baseline", "curv_contour_length_9", "curv_contour_length_14"]
    elif difficulty == "easy":
      sub_dirs = ["curv_baseline"]
    elif difficulty == "normal":
      sub_dirs = ["curv_contour_length_9"]
    elif difficulty == "hard":
      sub_dirs = ["curv_contour_length_14"]
    else:
      raise Exception(f"Unrecognized difficulty {difficulty}")

    # Get all image paths and their labels
    self.samples = []
    for sub_dir in sub_dirs:
      metadata_dir = os.path.join(data_root, data_dir, sub_dir, "metadata")
      for sample_group_file in os.listdir(metadata_dir):
        sample_group_dir = os.path.join(metadata_dir, sample_group_file)
        sample_group_file = open(sample_group_dir, "r")
        sample_group_lines = sample_group_file.readlines()[:-1]
        for sample_line in sample_group_lines:
          sample_tokens = sample_line[:-1].split(" ")
          sample_img_path = os.path.join(data_root, data_dir, sub_dir, sample_tokens[0], sample_tokens[1])
          sample_label = int(sample_tokens[3])
          self.samples.append((sample_img_path, sample_label))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    (img_path, label) = self.samples[idx]
    img = Image.open(open(img_path, "rb"))
    if self.transform is not None:
      img = self.transform(img)
    return (img, label)

  @staticmethod
  def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.stack([torch.tensor(item[1]).long() for item in batch])
    return (imgs, labels)

def pathfinder_32_loader(data_root, difficulty, batch_size, train_percentage):
  dataset = PathFinder32Dataset(data_root, difficulty=difficulty)
  num_train = int(len(dataset) * train_percentage)
  num_test = len(dataset) - num_train
  (train_dataset, test_dataset) = torch.utils.data.random_split(dataset, [num_train, num_test])
  train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=PathFinder32Dataset.collate_fn, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=PathFinder32Dataset.collate_fn, batch_size=batch_size, shuffle=True)
  return (train_loader, test_loader)

def build_adj(num_block_x, num_block_y):
  adjacency = []
  block_coord_to_block_id = lambda x, y: y * num_block_x + x
  for i, j in itertools.product(range(num_block_x), range(num_block_y)):
    for (dx, dy) in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
      x, y = i + dx, j + dy
      if x >= 0 and x < num_block_x and y >= 0 and y < num_block_y:
        source_id = block_coord_to_block_id(i, j)
        target_id = block_coord_to_block_id(x, y)
        adjacency.append((source_id, target_id))
  return adjacency

class PathFinderNet(nn.Module):
  def __init__(self, num_block_x=6, num_block_y=6):
    super(PathFinderNet, self).__init__()

    # block
    self.num_block_x = num_block_x
    self.num_block_y = num_block_y
    self.num_blocks = num_block_x * num_block_y
    self.block_coord_to_block_id = lambda x, y: y * num_block_x + x

    # Adjacency
    self.adjacency = build_adj(self.num_block_x, self.num_block_y)

    # CNN
    self.cnn = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=5),
      nn.Conv2d(32, 32, kernel_size=5),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, kernel_size=5),
      nn.Conv2d(64, 64, kernel_size=5),
      nn.MaxPool2d(2),
      nn.Flatten(),
    )

    # Fully connected for `is_endpoint`
    self.is_endpoint_fc = nn.Sequential(
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, self.num_blocks),
      nn.Sigmoid(),
    )

    # Fully connected for `connectivity`
    self.is_connected_fc = nn.Sequential(
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, len(self.adjacency)),
      nn.Sigmoid(),
    )

    # Exists Path
    self.last_fc = nn.Sequential(
      nn.Linear(self.num_blocks + len(self.adjacency), 1),
      nn.Sigmoid(),
    )

  def forward(self, image):
    embedding = self.cnn(image)
    is_connected = self.is_connected_fc(embedding) # 64 * 120
    is_endpoint = self.is_endpoint_fc(embedding) # 64 * 36
    feature = torch.cat((is_connected, is_endpoint), dim=1)
    result = self.last_fc(feature).reshape(-1)
    return result

class Trainer():
  def __init__(self, train_loader, test_loader, learning_rate, gpu, save_model=False):
    if gpu >= 0:
      device = torch.device("cuda:%d" % gpu)
    else:
      device = torch.device("cpu")
    self.device = device
    self.network = PathFinderNet.to(self.device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader

    # Aggregated loss (initialized to be a huge number)
    self.save_model = save_model
    self.min_test_loss = 100000000.0

  def loss_fn(self, output, ground_truth):
    return torch.mean(torch.square(output - ground_truth))

  def accuracy(self, output, expected_output) -> Tuple[int, int]:
    diff = torch.abs(output - expected_output)
    num_correct = len([() for d in diff if d.item() < 0.4999])
    return (len(output), num_correct)

  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    total_train_correct = 0
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for (input, expected_output) in iter:
      self.optimizer.zero_grad()
      input = input.to(self.device) # 64 * 1 * 32 * 32
      expected_output = expected_output.to(self.device)
      output = self.network(input)
      loss = self.loss_fn(output, expected_output)
      loss.backward()
      self.optimizer.step()
      batch_size, num_correct = self.accuracy(output, expected_output)
      num_items += batch_size
      total_train_correct += num_correct
      correct_perc = 100. * total_train_correct / num_items
      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}, Overall Accuracy: {correct_perc:.4f}%")

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = 0
    test_loss = 0
    total_correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (i, (input, expected_output)) in enumerate(iter):
        input = input.to(self.device)
        expected_output = expected_output.to(self.device)
        output = self.network(input)
        test_loss += self.loss_fn(output, expected_output).item()
        batch_size, num_correct_in_batch = self.accuracy(output, expected_output)
        num_items += batch_size
        total_correct += num_correct_in_batch
        perc = 100. * total_correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Avg loss: {test_loss / (i + 1):.4f}, Accuracy: {total_correct}/{num_items} ({perc:.2f}%)")

    # Save the model
    if self.save_model and test_loss < self.min_test_loss:
      self.min_test_loss = test_loss
      torch.save(self.network, "../model/pathfinder_32/pathfinder_32_net.pkl")

  def train(self, n_epochs):
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("pathfinder_32")
  parser.add_argument("--model-name", type=str, default="pathfinder.pkl")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--sample-count", type=int, default=100)
  parser.add_argument("--gpu", type=int, default=-1)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--train-percentage", type=float, default=0.9)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--difficulty", type=str, default="all")
  parser.add_argument("--cuda", action="store_true")
  args = parser.parse_args()

  # Setup parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  if args.cuda:
    if torch.cuda.is_available(): device = torch.device(f"cuda:{args.gpu}")
    else: raise Exception("No cuda available")
  else: device = torch.device("cpu")

  # Load data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../benchmarks/data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../model/pathfinder"))
  if not os.path.exists(model_dir): os.makedirs(model_dir)
  (train_loader, test_loader) = pathfinder_32_loader(data_dir, args.difficulty, args.batch_size, args.train_percentage)
  trainer = Trainer(train_loader, test_loader, args.learning_rate, args.gpu)

  # Run!
  trainer.train(args.n_epochs)