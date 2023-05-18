import torch

from constants import *
import strategy
import unstructured_dataset
import preprocess


class UnknownUnstructuredDataset(Exception):
    pass


class UnknownStructuredDataset(Exception):
    pass


class InvalidPreprocessStrategy(Exception):
    pass


class InvalidSampleStrategy(Exception):
    pass


class StructuredDataset:

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def collate_fn(batch, config):
        pass

    def forward(net, x):
        """
        Forwards the input `x` using the network `net`
        """
        pass

    def generate_datapoint(self):
        """
        Returns a sampled datapoint
        """
        pass

    def get_sample_strategy(self):
        """
        Returns a strategy object according to the sampling strategy specified 
        in the configuration
        """
        pass

    def preprocess_from_allowed_strategies(self, allowed):
        """
        Returns a preprocessing strategy object according to the strategy
        specified in the configuration. If the strategy is not in the `allowed`
        list, then raise `InvalidPreprocessStrategy`
        """
        s = self.config[PREPROCESS]
        if s not in allowed:
            raise InvalidPreprocessStrategy(
                "Preprocess strategy {s} is invalid")
        elif s == PREPROCESS_IDENTITY:
            strat = preprocess.PreprocessIdentity()
        elif s == PREPROCESS_SORT:
            strat = preprocess.PreprocessSort()
        return strat

    def get_preprocess_strategy(self):
        """
        Returns a preprocessing strategy object according to the strategy 
        specified in the configuration
        """

    def generate_dataset(self):
        """
        Returns a dataset of sampled datapoints
        """
        pass

    def flatten(config, input):
        """
        Returns a flattened list of input distributions.
        This is the format that is needed for the sampling algorithm to sample 
        inputs
        """
        pass

    def unflatten(config, samples, data, batch_item):
        """
        Returns an unflattened list of inputs.
        This is the format that is needed to pass the sampled inputs to a 
        black-box function
        """
        pass

    def n_unflatten(config):
        """
        Returns the length of each sublist of inputs to unflatten.
        This number is used to split the list of sampled inputs into lists of a 
        certain length. These lists are then unflattened to get a list of list
        of inputs to pass to the black-box function.
        """
        pass


class SingleDataset(StructuredDataset):

    def __init__(self, config, unstructured_dataset):
        self.config = config
        self.unstructured_dataset = unstructured_dataset
        self.strategy = self.get_sample_strategy()
        self.preprocess = self.get_preprocess_strategy()
        self.dataset = self.generate_dataset()

    def __len__(self):
        return len(self.unstructured_dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    @staticmethod
    def collate_fn(batch, config):
        return torch.stack(batch)

    def forward(net, x):
        return net(x)

    def generate_datapoint(self):
        return self.preprocess.preprocess(self.strategy.sample())

    def get_sample_strategy(self):
        s = self.config[STRATEGY]
        input_mapping = self.unstructured_dataset.input_mapping()
        if s == SINGLETON_STRATEGY:
            strat = strategy.SingletonStrategy(
                self.unstructured_dataset, input_mapping)
        else:
            raise InvalidSampleStrategy("Sampling strategy {s} is invalid")
        return strat

    def get_preprocess_strategy(self):
        allowed = [PREPROCESS_IDENTITY]
        return self.preprocess_from_allowed_strategies(allowed)

    def generate_dataset(self):
        length = self.__len__()
        dataset = [None] * length
        for i in range(length):
            dataset[i] = self.generate_datapoint()

    def flatten(config, input):
        return [input]

    def unflatten(config, samples, data, batch_item):
        return samples

    def n_unflatten(config):
        return 1


class IntDataset(StructuredDataset):

    def __init__(self, config, unstructured_dataset):
        self.config = config
        self.unstructured_dataset = unstructured_dataset
        self.strategy = self.get_sample_strategy()
        self.preprocess = self.get_preprocess_strategy()
        self.dataset = self.generate_dataset()

    def __len__(self):
        return int(len(self.unstructured_dataset) / self.config[N_DIGITS])

    def __getitem__(self, index):
        return self.dataset[index]

    @staticmethod
    def collate_fn(batch, config):
        imgs = [torch.stack([item[i] for item in batch])
                for i in range(len(batch[0]))]
        return imgs

    def forward(net, x):
        return [net(item) for item in x]

    def get_sample_strategy(self):
        n_digits = self.config[N_DIGITS]
        s = self.config[STRATEGY]
        input_mapping = [i for i in range(10)]
        if s == SIMPLE_LIST_STRATEGY:
            strat = strategy.SimpleListStrategy(
                self.unstructured_dataset, input_mapping, n_digits)
        else:
            raise InvalidSampleStrategy("Sampling strategy {s} is invalid")
        return strat

    def get_preprocess_strategy(self):
        allowed = [PREPROCESS_IDENTITY]
        return self.preprocess_from_allowed_strategies(allowed)

    def generate_datapoint(self):
        samples = self.preprocess.preprocess(self.strategy.sample())
        imgs, number_lst = zip(*samples)
        number = ''.join(str(n) for n in number_lst)
        return (imgs, int(number))

    def generate_dataset(self):
        length = self.__len__()
        dataset = [None] * length
        for i in range(length):
            dataset[i] = self.generate_datapoint()

    def flatten(config, input):
        return input

    def unflatten(config, samples, data, batch_item):
        number = ''
        for i in samples:
            number += str(i.item())
        return [torch.tensor(int(number))]

    def n_unflatten(config):
        return config[N_DIGITS]


class SingleIntListDataset(StructuredDataset):
    def __init__(self, config, unstructured_dataset):
        self.config = config
        self.unstructured_dataset = unstructured_dataset
        self.strategy = self.get_sample_strategy()
        self.preprocess = self.get_preprocess_strategy()
        self.dataset = self.generate_dataset()

    def __len__(self):
        return int(len(self.unstructured_dataset) / self.config[LENGTH])

    def __getitem__(self, index):
        return self.dataset[index]

    @staticmethod
    def collate_fn(batch, config):
        imgs = [torch.stack([item[i] for item in batch])
                for i in range(len(batch[0]))]
        return imgs

    def forward(net, x):
        return [net(item) for item in x]

    def get_sample_strategy(self):
        length = self.config[LENGTH]
        s = self.config[STRATEGY]
        input_mapping = [i for i in range(10)]
        if s == SIMPLE_LIST_STRATEGY:
            strat = strategy.SimpleListStrategy(
                self.unstructured_dataset, input_mapping, length)
        else:
            raise InvalidSampleStrategy("Sampling strategy {s} is invalid")
        return strat

    def get_preprocess_strategy(self):
        allowed = [PREPROCESS_IDENTITY, PREPROCESS_SORT]
        return self.preprocess_from_allowed_strategies(allowed)

    def generate_datapoint(self):
        samples = self.preprocess.preprocess(self.strategy.sample())
        return zip(*samples)

    def generate_dataset(self):
        length = self.__len__()
        dataset = [None] * length
        for i in range(length):
            dataset[i] = self.generate_datapoint()

    def flatten(config, input):
        return input

    def unflatten(config, samples, data, batch_item):
        return [samples]

    def n_unflatten(config):
        return config[LENGTH]


class IntListDataset(StructuredDataset):

    def __init__(self, config, unstructured_dataset):
        self.config = config
        self.unstructured_dataset = unstructured_dataset
        self.strategy = self.get_sample_strategy()
        self.preprocess = self.get_preprocess_strategy()
        self.dataset = self.generate_dataset()

    def __len__(self):
        return int(len(self.unstructured_dataset) / (self.config[N_DIGITS] * self.config[LENGTH]))

    def __getitem__(self, index):
        return self.dataset[index]

    @staticmethod
    def collate_fn(batch, config):
        return [[torch.stack([item[i][j] for item in batch]) for j in range(
            len(batch[0][0]))] for i in range(len(batch[0]))]

    def forward(net, x):
        return [[net(i) for i in item] for item in x]

    def get_sample_strategy(self):
        n_digits = self.config[N_DIGITS]
        s = self.config[STRATEGY]
        input_mapping = [i for i in range(10)]
        if s == SIMPLE_LIST_STRATEGY:
            strat = strategy.SimpleListStrategy(
                self.unstructured_dataset, input_mapping, n_digits)
        else:
            raise InvalidSampleStrategy("Sampling strategy {s} is invalid")
        return strat

    def get_preprocess_strategy(self):
        allowed = [PREPROCESS_IDENTITY, PREPROCESS_SORT]
        return self.preprocess_from_allowed_strategies(allowed)

    def generate_datapoint(self):
        lst = [None] * self.config[LENGTH]
        for i in range(self.config[LENGTH]):
            samples = self.strategy.sample()
            imgs, number_lst = zip(*samples)
            number = ''.join(str(n) for n in number_lst)
            lst[i] = (imgs, int(number))
        lst = self.preprocess.preprocess(lst)
        return zip(*lst)

    def generate_dataset(self):
        length = self.__len__()
        dataset = [None] * length
        for i in range(length):
            dataset[i] = self.generate_datapoint()

    def flatten(config, input):
        return [item for i in input for item in i]

    def unflatten(config, samples, data, batch_item):
        result = [0] * config[LENGTH]
        idx = 0
        for i in range(config[LENGTH]):
            number = ''
            for _ in range(config[N_DIGITS]):
                number += str(samples[idx].item())
                idx += 1
            result[i] = int(number)
        return [result]

    def n_unflatten(config):
        return config[LENGTH] * config[N_DIGITS]


class StringDataset(StructuredDataset):

    def __init__(self, config, unstructured_dataset):
        self.config = config
        self.unstructured_dataset = unstructured_dataset
        self.strategy = self.get_sample_strategy()
        self.preprocess = self.get_preprocess_strategy()
        self.dataset = self.generate_dataset()

    def __len__(self):
        return len(self.unstructured_dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    @staticmethod
    def collate_fn(batch, config):
        max_len = config[MAX_LENGTH]
        zero_img = torch.zeros_like(batch[0][0][0])

        def pad_zero(img_seq): return img_seq + \
            [zero_img] * (max_len - len(img_seq))
        img_seqs = torch.stack([torch.stack(pad_zero(img_seq))
                               for (img_seq, _) in batch])
        img_seq_len = torch.stack(
            [torch.tensor(img_seq_len).long() for (_, img_seq_len) in batch])
        return (img_seqs, img_seq_len)

    def forward(net, x):
        (distrs, _) = x
        return net(distrs)

    def get_sample_strategy(self):
        s = self.config[STRATEGY]
        input_mapping = [i for i in range(len(self.unstructured_dataset))]

        if s == SINGLETON_STRATEGY:
            strat = strategy.SingletonStrategy(
                self.unstructured_dataset, input_mapping)
        elif s == SIMPLE_LIST_STRATEGY:
            strat = strategy.SimpleListStrategy(
                self.unstructured_dataset, input_mapping, self.config[MAX_LENGTH]
            )
        else:
            raise InvalidSampleStrategy("Sampling strategy {s} is invalid")
        return strat

    def get_preprocess_strategy(self):
        allowed = [PREPROCESS_IDENTITY]
        return self.preprocess_from_allowed_strategies(allowed)

    def generate_datapoint(self):
        return self.preprocess.preprocess(self.strategy.sample())

    def generate_dataset(self):
        length = self.__len__()
        dataset = [None] * length
        for i in range(length):
            imgs, string = self.generate_datapoint()
            dataset[i] = ((imgs, len(string)), string)
        return dataset

    def flatten(config, input):
        return [item for item in torch.transpose(input, 0, 1)]

    def unflatten(config, samples, data, batch_item):
        length = data[1][batch_item].item()
        samples = samples[:length]
        ud = get_unstructured_dataset_static(config)
        input_mapping = ud.input_mapping(ud)
        string = ''
        for i in samples:
            string += input_mapping[i]
        return [string]

    def n_unflatten(config):
        return config[MAX_LENGTH]


def get_unstructured_dataset_static(config):
    ud = config[DATASET]
    if ud == MNIST:
        return unstructured_dataset.MNISTDataset
    elif ud == EMNIST:
        return unstructured_dataset.EMNISTDataset
    elif ud == HWF_SYMBOL:
        return unstructured_dataset.HWFDataset
    elif ud == MNIST_VIDEO:
        return unstructured_dataset.MNISTVideoDataset
    elif ud == MNIST_GRID:
        return unstructured_dataset.MNISTGridDataset
    else:
        raise UnknownUnstructuredDataset(f"Unknown dataset: {ud}")


def get_structured_dataset_static(config):
    sd = config[TYPE]
    if sd == DIGIT_TYPE:
        return SingleDataset
    elif sd == CHAR_TYPE:
        return SingleDataset
    elif sd == INT_TYPE:
        return IntDataset
    elif sd == SINGLE_INT_LIST_TYPE:
        return SingleIntListDataset
    elif sd == INT_LIST_TYPE:
        return IntListDataset
    elif sd == STRING_TYPE:
        return StringDataset
    else:
        raise UnknownStructuredDataset(f"Unknown dataset: {sd}")
