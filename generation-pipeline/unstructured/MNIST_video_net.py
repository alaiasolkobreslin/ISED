import torch


class View(torch.nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class MNISTVideoCNN(torch.nn.Module):
    def __init__(self, embedding_size=32):
        super().__init__()
        self.embedding_size = embedding_size
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=5),
            torch.nn.MaxPool2d(2),
            View(-1, 10816),
            torch.nn.Linear(10816, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, self.embedding_size),
            torch.nn.ReLU())
        self.digit_decoder = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, 10),
            torch.nn.Softmax(dim=1))
        self.changes_decoder = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size * 2, 1),
            torch.nn.Sigmoid())

    def forward(self, video_batch):
        (batch_size, num_frames, a, b, c) = video_batch.shape

        # First encode the video frames
        frame_encodings_batch = self.encoder(
            video_batch.view(batch_size * num_frames, a, b, c))

        # Predict the digits for each frame
        digits_batch = self.digit_decoder(
            frame_encodings_batch).view(batch_size, num_frames, -1)

        # Predict the changes for each consecutive pair of frames
        frame_encodings = frame_encodings_batch.view(
            batch_size, num_frames, self.embedding_size)
        frame_encodings_with_prepended_zero = torch.cat(
            [torch.zeros(batch_size, 1, self.embedding_size),
             frame_encodings[:, 0:num_frames - 1, :]],
            dim=1)
        consecutive_frame_encodings = torch.cat(
            [frame_encodings_with_prepended_zero, frame_encodings], dim=2)
        consecutive_frame_encodings_batch = consecutive_frame_encodings.view(
            batch_size * num_frames, -1)
        changes_batch = self.changes_decoder(
            consecutive_frame_encodings_batch).view(batch_size, num_frames)

        # Return
        return digits_batch, changes_batch
