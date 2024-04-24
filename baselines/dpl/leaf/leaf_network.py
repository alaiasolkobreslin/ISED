import torch
from torch import nn

class Leaf_Net(nn.Module):
  def __init__(self):
    super(Leaf_Net, self).__init__()

    # features for classification
    self.f1 = ['entire', 'indented', 'lobed', 'serrate', 'serrulate', 'undulate']
    self.f2 = ['elliptical', 'lanceolate', 'oblong', 'obovate', 'ovate']
    self.f3 = ['glossy', 'leathery', 'smooth', 'rough']
    self.labels = ['Alstonia Scholaris', 'Citrus limon', 'Jatropha curcas', 'Mangifera indica', 'Ocimum basilicum',
              'Platanus orientalis', 'Pongamia Pinnata', 'Psidium guajava', 'Punica granatum', 'Syzygium cumini', 'Terminalia Arjuna']
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

    # Fully connected for 'f2'
    self.f2_fc = nn.Sequential(
      nn.Linear(self.dim, self.dim),
      nn.ReLU(),
      nn.Linear(self.dim, len(self.f2)),
      nn.Softmax(dim=1)
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
    has_f1 = self.f1_fc(x)
    has_f2 = self.f2_fc(x)
    has_f3 = self.f3(x)
    return (has_f1, has_f2, has_f3)