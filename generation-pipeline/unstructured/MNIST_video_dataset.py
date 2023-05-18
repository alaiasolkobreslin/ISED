import json
import os

import cv2

import torch
import torchvision


class MNISTVideoDataset(torch.utils.data.Dataset):
    mnist_img_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    def __init__(self, root: str, filename: str, train: bool):
        # Load the metadata
        self.root = root
        self.label = "train" if train else "test"
        self.metadata = json.load(
            open(os.path.join(root, "data", filename)))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, i):
        dp = self.metadata[i]

        # Load video
        file_name = os.path.join(
            self.root, "data", "video", self.label, f"{dp['video_id']}.mp4")
        video = cv2.VideoCapture(file_name)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i in range(num_frames):
            ret, frame = video.read()
            frames.append(self.mnist_img_transform(frame)[0:1, :, :])
        video_tensor = torch.stack(frames)

        # Generate label 1, which is a list of digits occurred in the video
        label1 = []
        curr_image_id = None
        for frame in dp["frames_sg"]:
            if curr_image_id == None or curr_image_id != frame["image_id"]:
                curr_image_id = frame["image_id"]
                label1.append(frame["digit"])

        # Generate label 2, which is the list of digits for each frame
        label2 = torch.stack([torch.tensor(
            [1 if i == frame["digit"] else 0 for i in range(10)]) for frame in dp["frames_sg"]])

        # Generate label 3, which is the list of whether the image is different from the previous image in the video
        label3 = [1]
        for i in range(1, len(dp["frames_sg"])):
            label3.append(1 if dp["frames_sg"][i]["image_id"]
                          != dp["frames_sg"][i - 1]["image_id"] else 0)
        label3 = torch.tensor(label3)

        # Return the video and all the labels
        return (video_tensor, (label1, label2, label3))

    def collate_fn(batch):
        videos = torch.stack([item[0] for item in batch])
        label1 = [item[1][0] for item in batch]
        label2 = torch.stack([item[1][1]
                             for item in batch]).to(dtype=torch.float)
        label3 = torch.stack([item[1][2]
                             for item in batch]).to(dtype=torch.float)
        return (videos, (label1, label2, label3))

    def loaders(root, batch_size):
        train_loader = torch.utils.data.DataLoader(
            MNISTVideoDataset(root, "MNIST_video_train_1000.json", train=True),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=MNISTVideoDataset.collate_fn)
        test_loader = torch.utils.data.DataLoader(
            MNISTVideoDataset(root, "MNIST_video_test_10.json", train=False),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=MNISTVideoDataset.collate_fn)
        return train_loader, test_loader


def get_data(train):
    data_dir = os.path.abspath(os.path.join(
        os.path.abspath(__file__), "../../data/MNIST_video"))
    data = MNISTVideoDataset(
        data_dir, "MNIST_video_train_1000.json", train=train)
    return data
