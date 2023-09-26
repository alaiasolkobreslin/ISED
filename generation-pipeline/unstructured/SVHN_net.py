import torch.nn as nn
import torch.nn.functional as F


class SVHNNet(nn.Module):
    def __init__(self):
        super(SVHNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.conv4 = nn.Conv2d(128, 256, 5, 1, 2)
        self.fc1 = nn.Linear(2 * 2 * 256, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x_shape = x.shape
        if len(x_shape) == 5:
            x = x.view(-1, *x_shape[-3:])
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(*x_shape[:-3], 2 * 2 * 256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
