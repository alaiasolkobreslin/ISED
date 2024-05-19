from typing import Optional, Callable
import torch
import torchvision
from torch import nn, optim
from PIL import Image
import random
import os
import torch.nn.functional as F
import scallopy
from argparse import ArgumentParser
import time

torch.use_deterministic_algorithms(True)

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
      label = sample_group
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
    # labels = torch.stack([torch.tensor(item[1]).long() for item in batch])
    labels = [item[1] for item in batch]
    return (imgs, labels)

def leaves_loader(data_root, data_dir, batch_size, n_train, n_test):
  num_class = 11
  dataset = LeavesDataset(data_root, data_dir, (n_train+n_test))
  num_train = n_train*num_class
  num_test = len(dataset) - num_train
  (train_dataset, test_dataset) = torch.utils.data.random_split(dataset, [num_train, num_test])
  train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=LeavesDataset.collate_fn, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=LeavesDataset.collate_fn, batch_size=batch_size, shuffle=True)
  return (train_loader, test_loader)

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
    has_f3 = self.f3_fc(x)
    return has_f3
  
class Scallop_Leaf_Net(nn.Module):

  def __init__(self, provenance):
    super(Scallop_Leaf_Net, self).__init__()
    self.margin_net = Leaf_Net_Margin()
    self.texture_net = Leaf_Net_Texture()
    self.shape_net = Leaf_Net_Shape()
    self.base_ctx = scallopy.Context(provenance)
    self.base_ctx.import_file("scallop/leaves_config.scl")
    self.base_ctx.add_relation("margin", str, input_mapping=self.margin_net.f1)
    self.base_ctx.add_relation("shape", str, input_mapping=self.shape_net.f2)    
    self.base_ctx.add_relation("texture", str, input_mapping=self.texture_net.f3)
    self.predict = self.base_ctx.forward_function("get_prediction", dispatch="parallel")

  def forward(self, x):
    margins = self.margin_net(x)
    textures = self.texture_net(x)
    shapes = self.shape_net(x)
    r =  self.predict(margin = margins, texture = textures, shape = shapes)
    return r

class Trainer():
  def __init__(self, train_loader, test_loader, device, model_root, learning_rate, provenance, phase = "train"):
    self.device = device
    # if not model_root == None and os.path.exists(model_root + '.best.pt'):
    #   self.network = torch.load(open(model_root + '.best.pt', 'rb'))
    # else:
    self.network = Scallop_Leaf_Net(provenance).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.loss_fn = F.binary_cross_entropy
    self.model_root = model_root
    self.min_test_loss = 100000000.0
    self.phase = phase
    self.max_test_acc = 0

  def _loss_fn(self, y_pred_values, y_pred_probs, y_values):
    y = torch.stack([torch.tensor([1.0 if str(u[0]) == str(v) else 0.0 for u in y_pred_values]) for v in y_values]).to(self.device)
    return self.loss_fn(y_pred_probs, y)

  def _num_correct(self, y_pred_values, y_pred_probs, y_values):
    indices = torch.argmax(y_pred_probs, dim=1).to("cpu")
    predicted = [y_pred_values[i][0] for i in indices]
    return sum([1 if str(x) == str(y) else 0 for (x, y) in zip(predicted, y_values)])

  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    train_loss = 0
    total_correct = 0
    for (i, (x, y)) in enumerate(train_loader):
      batch_size = len(y)
      x = x.to(device)
      # Do the prediction and obtain the loss/accuracy
      (y_pred_values, y_pred_probs) = self.network(x)
      loss = self._loss_fn(y_pred_values, y_pred_probs, y)
      num_correct = self._num_correct(y_pred_values, y_pred_probs, y)

      # Compute loss
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      # Stats
      train_loss += loss
      num_items += batch_size
      total_correct += num_correct
      perc = 100. * total_correct / num_items
      avg_loss = train_loss / (i + 1)

      # Prints
    print(f"[Train Epoch {epoch}] Avg loss: {avg_loss:.4f}, Accuracy: {total_correct}/{num_items} ({perc:.2f}%)")
    torch.save(self.network, self.model_root + '/latest.pt')
    return train_loss

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = 0
    test_loss = 0
    total_correct = 0
    with torch.no_grad():
      for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        batch_size = len(y)

        # Do the prediction and obtain the loss/accuracy
        (y_pred_values, y_pred_probs) = self.network(x)
        loss = self._loss_fn(y_pred_values, y_pred_probs, y)
        num_correct = self._num_correct(y_pred_values, y_pred_probs, y)

        # Stats
        test_loss += loss
        num_items += batch_size
        total_correct += num_correct
        perc = 100. * total_correct / num_items
        avg_loss = test_loss / (i + 1)

        self.max_test_acc = max(self.max_test_acc, perc)

        # Prints
      print(f"[Test Epoch {epoch}] Avg loss: {avg_loss:.4f}, Accuracy: {total_correct}/{num_items} ({perc:.2f}%)")

    # Save model
    if test_loss < self.min_test_loss and self.phase == "train":
      self.min_test_loss = test_loss
      torch.save(self.network, self.model_root + '/best.pt')

    return float(total_correct / num_items)

  def train(self, n_epochs):
    for epoch in range(1, n_epochs + 1):
      t0 = time.time()
      train_loss = self.train_epoch(epoch)
      t1 = time.time()
      test_acc = self.test_epoch(epoch)

  def test(self):
    self.test_epoch(0)


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sum_2")
  parser.add_argument("--n-epochs", type=int, default=50)
  parser.add_argument("--batch-size-train", type=int, default=16)
  parser.add_argument("--batch-size-test", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--top-k", type=int, default=3)
  parser.add_argument("--jit", action="store_true")
  parser.add_argument("--dispatch", type=str, default="parallel")
  parser.add_argument("--device", type=str, default="cpu")
  args = parser.parse_args()

  n_epochs = args.n_epochs
  batch_size_train = args.batch_size_train
  batch_size_test = args.batch_size_test
  learning_rate = args.learning_rate
  loss_fn = args.loss_fn
  k = args.top_k
  provenance = args.provenance
  device = args.device
  seed = args.seed

  torch.manual_seed(seed)
  random.seed(seed)
    
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
  model_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../"))
    
  train_loader, test_loader = leaves_loader(data_root, "leaf_11", batch_size_train, 30, 10)
  trainer = Trainer(train_loader=train_loader, test_loader=test_loader, learning_rate=learning_rate, device=device, model_root=model_root, provenance=provenance)
  trainer.train(n_epochs)