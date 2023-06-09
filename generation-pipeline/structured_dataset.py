import torch

from functools import partial

from constants import *
import strategy
import unstructured_dataset
import preprocess
import input


def id(x):
    return x


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

    def get_input_mapping(config):
        """
        Returns the input mapping for the structured dataset
        """
        pass

    def distrs_to_input(distrs, x, config):
        """
        Returns the distrs in a cleaner input format (such as ListInput), if
        required. The original input `x` is also given in case input lengths
        are needed such as in padded inputs.
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

    def get_input_mapping(config):
        ud = get_unstructured_dataset_static(config)
        return input.DiscreteInputMapping(ud.input_mapping(ud), id)

    def distrs_to_input(distrs, x, config):
        return distrs


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
    def collate_fn(batch, _):
        return torch.stack([torch.stack(item) for item in batch])

    def forward(net, x):
        batch_size, length, _, _, _ = x.shape
        return net(x.flatten(start_dim=0, end_dim=1)).view(batch_size, length, -1)

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

    def combine(inputs):
        return int("".join(str(s) for s in inputs))

    def get_input_mapping(config):
        ud = get_unstructured_dataset_static(config)
        length = config[N_DIGITS]
        element_input_mapping = input.DiscreteInputMapping(
            ud.input_mapping(ud), id)
        return input.ListInputMapping(length, element_input_mapping, IntDataset.combine)

    def distrs_to_input(distrs, x, config):
        length = config[N_DIGITS]
        return input.ListInput(distrs, length)


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
    def collate_fn(batch, _):
        return torch.stack([torch.stack(item) for item in batch])

    def forward(net, x):
        batch_size, length, _, _, _ = x.shape
        return net(x.flatten(start_dim=0, end_dim=1)).view(batch_size, length, -1)

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

    def get_input_mapping(config):
        ud = get_unstructured_dataset_static(config)
        length = config[LENGTH]
        element_input_mapping = input.DiscreteInputMapping(
            ud.input_mapping(ud), id)
        return input.ListInputMapping(length, element_input_mapping, id)

    def distrs_to_input(distrs, x, config):
        length = config[LENGTH]
        return input.ListInput(distrs, length)


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
        return torch.stack([torch.stack([torch.stack(i) for i in item]) for item in batch])

    def forward(net, x):
        batch_size, length, n_digits, _, _, _ = x.shape
        return net(x.flatten(start_dim=0, end_dim=2)).view(batch_size, length, n_digits, -1)

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

    def combine(n_digits, input):
        result = []
        i = 0
        current_int = ""
        while i < len(input):
            current_int += str(input[i])
            i += 1
            if i % n_digits == 0:
                result.append(int(current_int))
                current_int = ""
        return result

    def get_input_mapping(config):
        ud = get_unstructured_dataset_static(config)
        length = config[LENGTH]
        n_digits = config[N_DIGITS]
        digit_input_mapping = input.DiscreteInputMapping(
            ud.input_mapping(ud), id)
        return input.ListInputMapping2D(length, n_digits, digit_input_mapping, partial(IntListDataset.combine, n_digits))

    def distrs_to_input(distrs, x, config):
        n_rows = config[LENGTH]
        n_cols = config[N_DIGITS]
        return input.ListInput2D(distrs, n_rows, n_cols)


class SingleIntListListDataset(StructuredDataset):
    pass


class SingleIntGridDataset(StructuredDataset):
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
        return torch.stack([torch.stack([torch.stack(i) for i in item]) for item in batch])

    def forward(net, x):
        batch_size, length, n_digits, _, _, _ = x.shape
        return net(x.flatten(start_dim=0, end_dim=2)).view(batch_size, length, n_digits, -1)

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
        lst = [None] * self.config[LENGTH]
        for i in range(self.config[LENGTH]):
            samples = self.strategy.sample()
            imgs, row = zip(*samples)
            lst[i] = (imgs, tuple(row))
        lst = self.preprocess.preprocess(lst)
        return zip(*lst)

    def generate_dataset(self):
        length = self.__len__()
        dataset = [None] * length
        for i in range(length):
            dataset[i] = self.generate_datapoint()

    def combine(length, input):
        result = []
        i = 0
        current_row = []
        while i < len(input):
            current_row.append(input[i])
            i += 1
            if i % length == 0:
                result.append(current_row)
                current_row = []
        return result

    def get_input_mapping(config):
        ud = get_unstructured_dataset_static(config)
        length = config[LENGTH]
        digit_input_mapping = input.DiscreteInputMapping(
            ud.input_mapping(ud), id)
        return input.ListInputMapping2D(length, length, digit_input_mapping, partial(SingleIntGridDataset.combine, length))

    def distrs_to_input(distrs, x, config):
        length = config[LENGTH]
        return input.ListInput2D(distrs, length, length)


class PaddedStringDataset(StructuredDataset):

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

    def combine(inputs):
        return "".join(str(s) for s in inputs)

    def get_input_mapping(config):
        max_length = config[MAX_LENGTH]
        ud = get_unstructured_dataset_static(config)
        element_input_mapping = input.DiscreteInputMapping(
            ud.input_mapping(ud), id)
        return input.PaddedListInputMapping(max_length, element_input_mapping, PaddedStringDataset.combine)

    def distrs_to_input(distrs, x, config):
        lengths = [l.item() for l in x[1]]
        return input.PaddedListInput(distrs, lengths)


class StringDataset(StructuredDataset):
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
        imgs, string_list = zip(*samples)
        string = ''.join(str(n) for n in string_list)
        return (imgs, string)

    def generate_dataset(self):
        length = self.__len__()
        dataset = [None] * length
        for i in range(length):
            dataset[i] = self.generate_datapoint()

    def combine(inputs):
        return "".join(str(s) for s in inputs)

    def get_input_mapping(config):
        length = config[LENGTH]
        ud = get_unstructured_dataset_static(config)
        element_input_mapping = input.DiscreteInputMapping(
            ud.input_mapping(ud), StringDataset.combine, id)
        return input.ListInputMapping(length, element_input_mapping, StringDataset.combine)

    def distrs_to_input(distrs, x, config):
        length = config[LENGTH]
        return input.ListInput(distrs, length)


def get_unstructured_dataset_static(config):
    ud = config[DATASET]
    if ud == MNIST:
        return unstructured_dataset.MNISTDataset
    elif ud == EMNIST:
        return unstructured_dataset.EMNISTDataset
    elif ud == SVHN:
        return unstructured_dataset.SVHNDataset
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
    elif sd == SINGLE_INT_LIST_LIST_TYPE:
        return SingleIntListListDataset
    elif sd == SINGLE_INT_GRID_TYPE:
        return SingleIntGridDataset
    elif sd == INT_LIST_TYPE:
        return IntListDataset
    elif sd == STRING_TYPE and MAX_LENGTH in config:
        return PaddedStringDataset
    elif sd == STRING_TYPE:
        return StringDataset
    else:
        raise UnknownStructuredDataset(f"Unknown dataset: {sd}")
