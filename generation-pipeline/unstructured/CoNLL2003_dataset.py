from transformers import BertTokenizerFast
from datasets import load_dataset

import torch


class Aligner:

    def __init__(self, labels_to_ids):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        self.labels_to_ids = labels_to_ids

    def align_label(self, texts, labels):
        tokenized_inputs = self.tokenizer(
            texts, padding='max_length', max_length=512, truncation=True)

        word_ids = tokenized_inputs.word_ids()

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                try:
                    label_ids.append(self.labels_to_ids[labels[word_idx]])
                except:
                    label_ids.append(-100)
            else:
                try:
                    label_ids.append(
                        self.labels_to_ids[labels[word_idx]] if self.label_all_tokens else -100)
                except:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        return label_ids


class DataSequence(torch.utils.data.Dataset):

    def __init__(self, dataset):
        labels = [sentence['ner_tags'] for sentence in dataset]
        self.unique_labels = set()
        for lb in labels:
            [self.unique_labels.add(i)
             for i in lb if i not in self.unique_labels]
        labels_to_ids = {k: v for v, k in enumerate(
            sorted(self.unique_labels))}
        ids_to_labels = {v: k for v, k in enumerate(
            sorted(self.unique_labels))}
        aligner = Aligner(labels_to_ids)
        txt = [sentence['tokens'] for sentence in dataset]
        self.texts = [aligner.tokenizer(str(i),
                                        padding='max_length', max_length=512, truncation=True, return_tensors="pt") for i in txt]
        self.labels = [aligner.align_label(i, j) for i, j in zip(txt, labels)]

    def __len__(self):

        return len(self.labels)

    def get_batch_data(self, idx):

        return self.texts[idx]

    def get_batch_labels(self, idx):

        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):

        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels


def get_data(train: bool):
    dataset = load_dataset("conll2003")
    split = 'train' if train else 'test'
    data = dataset[split]
    data_sequence = DataSequence(data)

    return data_sequence
