import torch


class View(torch.nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class MNISTVideoLSTM(torch.nn.Module):
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
            torch.nn.Linear(1024, self.embedding_size))
        self.lstm = torch.nn.LSTM(
            self.embedding_size, self.embedding_size, batch_first=True)
        self.digit_decoder = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, 10),
            torch.nn.Softmax(dim=1))
        self.changes_decoder = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, 1),
            torch.nn.Sigmoid())

    def forward(self, video_batch):
        (batch_size, num_frames, a, b, c) = video_batch.shape
        frame_encodings_batch = self.encoder(
            video_batch.view(batch_size * num_frames, a, b, c))
        frame_embeddings_batch_raw, _ = self.lstm(frame_encodings_batch)
        frame_embeddings_batch = frame_embeddings_batch_raw.view(
            batch_size * num_frames, -1)
        digits_batch = self.digit_decoder(
            frame_embeddings_batch).view(batch_size, num_frames, -1)
        changes_batch = self.changes_decoder(frame_embeddings_batch).view(
            batch_size, num_frames, -1).view(batch_size, num_frames)
        return digits_batch, changes_batch
