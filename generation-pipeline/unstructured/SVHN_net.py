import torch.nn as nn
import torch.nn.functional as F


class SVHNNet(nn.Module):
    def __init__(self):
        super(SVHNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=6)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=6)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
