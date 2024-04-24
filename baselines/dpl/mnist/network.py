import torch.nn as nn
import torch.nn.functional as F


# class MNIST_CNN(nn.Module):
#     def __init__(self):
#         super(MNIST_CNN, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 6, 5),
#             nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
#             nn.ReLU(True),
#             nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
#             nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
#             nn.ReLU(True),
#         )

#     def forward(self, x):
#         x = x.unsqueeze(0)
#         x = self.encoder(x)
#         x = x.view(-1, 16 * 4 * 4)
#         return x


# class MNIST_Classifier(nn.Module):
#     def __init__(self, n=10, with_softmax=True):
#         super(MNIST_Classifier, self).__init__()
#         self.with_softmax = with_softmax
#         if with_softmax:
#             self.softmax = nn.Softmax(1)
#         self.classifier = nn.Sequential(
#             nn.Linear(16 * 4 * 4, 120),
#             nn.ReLU(),
#             nn.Linear(120, 84),
#             nn.ReLU(),
#             nn.Linear(84, n),
#         )

#     def forward(self, x):
#         x = self.classifier(x)
#         if self.with_softmax:
#             x = self.softmax(x)
#         return x.squeeze(0)

class MNIST_Net(nn.Module):
    def __init__(self, n_preds):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, n_preds)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


# class MNIST_Net(nn.Module):
#     def __init__(self, n=10, with_softmax=True, size=16 * 4 * 4):
#         super(MNIST_Net, self).__init__()
#         self.with_softmax = with_softmax
#         self.size = size
#         if with_softmax:
#             if n == 1:
#                 self.softmax = nn.Sigmoid()
#             else:
#                 self.softmax = nn.Softmax(1)
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 6, 5),
#             nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
#             nn.ReLU(True),
#             nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
#             nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
#             nn.ReLU(True),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(size, 120),
#             nn.ReLU(),
#             nn.Linear(120, 84),
#             nn.ReLU(),
#             nn.Linear(84, n),
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = x.view(-1, self.size)
#         x = self.classifier(x)
#         if self.with_softmax:
#             x = self.softmax(x)
#         return x
