import os
from tqdm import tqdm
from argparse import ArgumentParser
import json
from PIL import Image
import torch.optim as optim
import torchvision
from typing import *
import random
import torch.nn as nn
import torch.nn.functional as F
import torch

coffee_img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.Normalize(
        (0.1307,), (0.3081,)
    )
])


class CoffeeBaselineDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        prefix: str,
        train: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.prefix = prefix
        self.root = root
        self.transform = transform
        self.split = 'train_leaves' if train else 'test_leaves'
        self.img_dir = 'miner_img_xml' if prefix == 'miner' else 'rust_xml_image'
        self.metadata = json.load(
            open(os.path.join(root, f"Coffee_leaf/{prefix}_examples.json")))[self.split]
        self.area_dict = json.load(
            open(os.path.join(root, f"Coffee_leaf/{prefix}_areas.json")))
        self.quantiles = self.area_dict['train_quantiles']
        self.dataset_length = len(self.metadata)
        self.index_map = list(range(self.dataset_length))
        random.shuffle(self.index_map)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        # Get image
        leaf = self.metadata[self.index_map[idx]]
        leaf_path = os.path.join(
            self.root, f"Coffee_leaf/{self.img_dir}/{leaf}")
        img_full_path = leaf_path + ".jpg"
        img = Image.open(img_full_path).convert("RGB")

        # transform image
        if self.transform:
            img = self.transform(img)

        # Get GT severity
        affected_area = self.area_dict[self.split][leaf]
        severity = self.get_severity_score(affected_area)

        return (img, severity)

    @staticmethod
    def collate_fn(batch):
        imgs = torch.stack([item[0] for item in batch])
        severity_scores = torch.stack(
            [torch.tensor(item[1]).long() for item in batch])
        return (imgs, severity_scores)

    def get_severity_score(self, area):
        if area < self.quantiles['Q1']:
            return 1
        elif area < self.quantiles['Q2']:
            return 2
        elif area < self.quantiles['Q3']:
            return 3
        elif area < self.quantiles['Q4']:
            return 4
        else:
            return 5


def coffee_baseline_loader(data_dir, prefix, batch_size_train, batch_size_test):
    train_loader = torch.utils.data.DataLoader(
        CoffeeBaselineDataset(
            data_dir,
            prefix,
            train=True,
            transform=coffee_img_transform,
        ),
        collate_fn=CoffeeBaselineDataset.collate_fn,
        batch_size=batch_size_train,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        CoffeeBaselineDataset(
            data_dir,
            prefix,
            train=False,
            transform=coffee_img_transform,
        ),
        collate_fn=CoffeeBaselineDataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=True
    )

    return train_loader, test_loader


class CoffeeNet(nn.Module):
    def __init__(self):
        super(CoffeeNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)  # Output classes: 0 or 1

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class Trainer():
    def __init__(self, train_loader, test_loader, learning_rate):
        self.network = CoffeeNet()
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=learning_rate)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.best_loss = 10000000000

    def loss(self, output, ground_truth):
        (_, dim) = output.shape
        gt = torch.stack([torch.tensor(
            [1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
        return F.binary_cross_entropy(output, gt)

    def train_epoch(self, epoch):
        self.network.train()
        iter = tqdm(self.train_loader, total=len(self.train_loader))
        for (data, target) in iter:
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            iter.set_description(
                f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")

    def test_epoch(self, epoch):
        self.network.eval()
        num_items = len(self.test_loader.dataset)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            iter = tqdm(self.test_loader, total=len(self.test_loader))
            for (data, target) in iter:
                output = self.network(data)
                test_loss += self.loss(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                perc = 100. * correct / num_items
                iter.set_description(
                    f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")
            if test_loss < self.best_loss:
                self.best_loss = test_loss

    def train(self, n_epochs):
        self.test_epoch(0)
        for epoch in range(1, n_epochs + 1):
            self.train_epoch(epoch)
            self.test_epoch(epoch)


if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("mnist_sum_2")
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size-train", type=int, default=64)
    parser.add_argument("--batch-size-test", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--prefix", type=str, default='miner')
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    # Parameters
    n_epochs = args.n_epochs
    batch_size_train = args.batch_size_train
    batch_size_test = args.batch_size_test
    learning_rate = args.learning_rate
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Data
    data_dir = os.path.abspath(os.path.join(
        os.path.abspath(__file__), "../../data"))

    # Dataloaders
    train_loader, test_loader = coffee_baseline_loader(
        data_dir, args.prefix, batch_size_train, batch_size_test)

    # Create trainer and train
    trainer = Trainer(train_loader, test_loader,
                      learning_rate)
    trainer.train(n_epochs)
