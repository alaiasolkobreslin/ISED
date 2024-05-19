import torch
import torch.nn as nn
import torch.nn.functional as F

class _LeafNet(nn.Module):
  def __init__(self, num_features, dim):
    super(_LeafNet, self).__init__()
    self.N = num_features
    self.size = dim

    # CNN
    self.encoder = nn.Sequential(
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
    )

    # Fully connected for 'features'
    self.classifier = nn.Sequential(
      nn.Linear(self.size, self.size),
      nn.ReLU(),
      nn.Linear(self.size, self.N),
    )

    self.softmax = nn.Softmax(dim=1)
    
  def forward(self, x):
    x = self.encoder(x)
    x = x.view(-1, self.size)
    x = self.classifier(x)   
    x = self.softmax(x)
    return x  
  
class LeafNet(nn.Module):
  def __init__(self):
    super(LeafNet, self).__init__()
    self.net1 = _LeafNet(6, 2304)
    self.net2 = _LeafNet(5, 2304)
    self.net3 = _LeafNet(4, 2304)

  def forward(self, x):
    has_f1 = self.net1(x)
    has_f2 = self.net2(x)
    has_f3 = self.net3(x)
    return (has_f1, has_f2, has_f3)

class LLMLeafNet(nn.Module):
  def __init__(self):
    super(LLMLeafNet, self).__init__()
    self.net1 = _LeafNet(3, 2304)
    self.net2 = _LeafNet(6, 2304)
    self.net3 = _LeafNet(3, 2304)

  def forward(self, x):
    has_f1 = self.net1(x)
    has_f2 = self.net2(x)
    has_f3 = self.net3(x)
    return (has_f1, has_f2, has_f3)