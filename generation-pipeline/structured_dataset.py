import torch
import numpy as np

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

    def __init__(self, config, dataset_size, unstructured_dataset, call_black_box_for_gt):
        self.config = config
        self.dataset_size = dataset_size
        self.unstructured_dataset = unstructured_dataset
        self.im = self.unstructured_dataset.input_mapping()
        self.strategy = self.get_sample_strategy()
        self.preprocess = self.get_preprocess_strategy()
        self.call_black_box_for_gt = call_black_box_for_gt

    def __len__(self):
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
                f"Preprocess strategy {s} is invalid")
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

    def __init__(self, config, dataset_size, unstructured_dataset):
        super().__init__(config=config,
                         dataset_size=dataset_size,
                         unstructured_dataset=unstructured_dataset,
                         call_black_box_for_gt=True)

    def __len__(self):
        return self.dataset_size

    @staticmethod
    def collate_fn(batch, config):
        return torch.stack(batch)

    def forward(net, x):
        return net(x)

    def generate_datapoint(self):
        (ud, sd) = self.strategy.sample()
        return self.preprocess.preprocess((ud, self.im[sd]))

    def get_sample_strategy(self):
        s = self.config[STRATEGY]
        if s == SINGLE_SAMPLE_STRATEGY:
            strat = strategy.SingleSampleStrategy(
                self.unstructured_dataset, self.im)
        else:
            raise InvalidSampleStrategy(f"Sampling strategy {s} is invalid")
        return strat

    def get_preprocess_strategy(self):
        allowed = [PREPROCESS_IDENTITY]
        return self.preprocess_from_allowed_strategies(allowed)

    def get_input_mapping(config):
        ud = get_unstructured_dataset_static(config)
        return input.DiscreteInputMapping(ud.input_mapping(ud), id)

    def distrs_to_input(distrs, x, config):
        return input.SingleInput(distrs)


class IntDataset(StructuredDataset):

    def __init__(self, config, dataset_size, unstructured_dataset):
        super().__init__(config=config,
                         dataset_size=dataset_size,
                         unstructured_dataset=unstructured_dataset,
                         call_black_box_for_gt=True)

    def __len__(self):
        return self.dataset_size

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
            raise InvalidSampleStrategy(f"Sampling strategy {s} is invalid")
        return strat

    def get_preprocess_strategy(self):
        allowed = [PREPROCESS_IDENTITY, PREPROCESS_PALINDROME]
        return self.preprocess_from_allowed_strategies(allowed)

    def generate_datapoint(self):
        samples = self.preprocess.preprocess(self.strategy.sample())
        imgs, number_lst = zip(*samples)
        number = ''.join(str(n) for n in number_lst)
        return (imgs, int(number))

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
    def __init__(self, config, dataset_size, unstructured_dataset):
        super().__init__(config=config,
                         dataset_size=dataset_size,
                         unstructured_dataset=unstructured_dataset,
                         call_black_box_for_gt=True)

    def __len__(self):
        return self.dataset_size

    @staticmethod
    def collate_fn(batch, _):
        return torch.stack([torch.stack(item) for item in batch])

    def forward(net, x):
        batch_size, length, _, _, _ = x.shape
        return net(x.flatten(start_dim=0, end_dim=1)).view(batch_size, length, -1)

    def get_sample_strategy(self):
        s = self.config[STRATEGY]
        if s == SINGLE_SAMPLE_STRATEGY:
            input_mapping = [i for i in range(len(self.unstructured_dataset))]
            strat = strategy.SingleSampleStrategy(
                self.unstructured_dataset, input_mapping)
        elif s == SIMPLE_LIST_STRATEGY:
            length = self.config[LENGTH]
            strat = strategy.SimpleListStrategy(
                self.unstructured_dataset, self.im, length)
        else:
            raise InvalidSampleStrategy(f"Sampling strategy {s} is invalid")
        return strat

    def get_preprocess_strategy(self):
        allowed = [PREPROCESS_IDENTITY, PREPROCESS_SORT]
        return self.preprocess_from_allowed_strategies(allowed)

    def generate_datapoint(self):
        samples = self.preprocess.preprocess(self.strategy.sample())
        return zip(*samples)

    def combine(config, input):
        # TODO: fix this. It's kind of a hacky way to do preprocessing
        if config[PREPROCESS] == PREPROCESS_SORT:
            return sorted(input)
        else:
            return input

    def get_input_mapping(config):
        ud = get_unstructured_dataset_static(config)
        length = config[LENGTH]
        element_input_mapping = input.DiscreteInputMapping(
            ud.input_mapping(ud), id)
        return input.ListInputMapping(length, element_input_mapping, partial(SingleIntListDataset.combine, config))

    def distrs_to_input(distrs, x, config):
        length = config[LENGTH]
        return input.ListInput(distrs, length)


class IntListDataset(StructuredDataset):

    def __init__(self, config, dataset_size, unstructured_dataset):
        super().__init__(config=config,
                         dataset_size=dataset_size,
                         unstructured_dataset=unstructured_dataset,
                         call_black_box_for_gt=True)

    def __len__(self):
        return self.dataset_size

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
            raise InvalidSampleStrategy(f"Sampling strategy {s} is invalid")
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


class StringListDataset(StructuredDataset):

    def __init__(self, config, dataset_size, unstructured_dataset):
        super().__init__(config=config,
                         dataset_size=dataset_size,
                         unstructured_dataset=unstructured_dataset,
                         call_black_box_for_gt=True)

    def __len__(self):
        return self.dataset_size

    @staticmethod
    def collate_fn(batch, config):
        return torch.stack([torch.stack([torch.stack(i) for i in item]) for item in batch])

    def forward(net, x):
        batch_size, lst_length, str_length, _, _, _ = x.shape
        return net(x.flatten(start_dim=0, end_dim=2)).view(batch_size, lst_length, str_length, -1)

    def get_sample_strategy(self):
        str_length = self.config[STR_LENGTH]
        s = self.config[STRATEGY]
        if s == SIMPLE_LIST_STRATEGY:
            strat = strategy.SimpleListStrategy(
                self.unstructured_dataset, self.im, str_length)
        else:
            raise InvalidSampleStrategy(f"Sampling strategy {s} is invalid")
        return strat

    def get_preprocess_strategy(self):
        allowed = [PREPROCESS_IDENTITY]
        return self.preprocess_from_allowed_strategies(allowed)

    def generate_datapoint(self):
        lst = [None] * self.config[LENGTH]
        for i in range(self.config[LENGTH]):
            samples = self.strategy.sample()
            imgs, chr_lst = zip(*samples)
            s = ''.join(str(self.im[n]) for n in chr_lst)
            lst[i] = (imgs, s)
        lst = self.preprocess.preprocess(lst)
        return zip(*lst)

    def combine(str_length, input):
        result = []
        i = 0
        current_str = ""
        while i < len(input):
            current_str += str(input[i])
            i += 1
            if i % str_length == 0:
                result.append(current_str)
                current_str = ""
        return result

    def get_input_mapping(config):
        ud = get_unstructured_dataset_static(config)
        length = config[LENGTH]
        str_length = config[STR_LENGTH]
        chr_input_mapping = input.DiscreteInputMapping(
            ud.input_mapping(ud), id)
        return input.ListInputMapping2D(length, str_length, chr_input_mapping, partial(StringListDataset.combine, str_length))

    def distrs_to_input(distrs, x, config):
        n_rows = config[LENGTH]
        n_cols = config[STR_LENGTH]
        return input.ListInput2D(distrs, n_rows, n_cols)


class SingleIntListListDataset(StructuredDataset):
    def __init__(self, config, dataset_size, unstructured_dataset):
        super().__init__(config=config,
                         dataset_size=dataset_size,
                         unstructured_dataset=unstructured_dataset,
                         call_black_box_for_gt=True)

    def __len__(self):
        return self.dataset_size

    @staticmethod
    def collate_fn(batch, config):
        return torch.stack([torch.stack([torch.stack(i) for i in item]) for item in batch])

    def forward(net, x):
        batch_size, length, n_digits, _, _, _ = x.shape
        return net(x.flatten(start_dim=0, end_dim=2)).view(batch_size, length, n_digits, -1)

    def get_sample_strategy(self):
        n_rows = self.config[N_ROWS]
        n_cols = self.config[N_COLS]
        s = self.config[STRATEGY]
        input_mapping = [i for i in range(10)]
        if s == LIST_2D:
            strat = strategy.Simple2DListStrategy(
                self.unstructured_dataset, input_mapping, n_rows, n_cols)
        else:
            raise InvalidSampleStrategy(f"Sampling strategy {s} is invalid")
        return strat

    def get_preprocess_strategy(self):
        allowed = [PREPROCESS_IDENTITY]
        return self.preprocess_from_allowed_strategies(allowed)

    def generate_datapoint(self):
        samples = self.strategy.sample()
        samples = self.preprocess.preprocess(samples)
        return zip(*samples)

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
        n_rows = config[N_ROWS]
        n_cols = config[N_COLS]
        digit_input_mapping = input.DiscreteInputMapping(
            ud.input_mapping(ud), id)
        return input.ListInputMapping2D(n_rows, n_cols, digit_input_mapping, partial(SingleIntListListDataset.combine, n_cols))

    def distrs_to_input(distrs, x, config):
        n_rows = config[N_ROWS]
        n_cols = config[N_COLS]
        return input.ListInput2D(distrs, n_rows, n_cols)


class PaddedListDataset(StructuredDataset):

    def __init__(self, config, dataset_size, unstructured_dataset):
        super().__init__(config=config,
                         dataset_size=dataset_size,
                         unstructured_dataset=unstructured_dataset,
                         call_black_box_for_gt=True)

    def __len__(self):
        return self.dataset_size

    @staticmethod
    def collate_fn(batch, config):
        max_len = config[MAX_LENGTH]
        zero_img = torch.zeros_like(batch[0][0][0], device=DEVICE)

        def pad_zero(img_seq): return img_seq + \
            [zero_img] * (max_len - len(img_seq))
        img_seqs = torch.stack([torch.stack(pad_zero(img_seq))
                               for (img_seq, _) in batch])
        img_seq_len = torch.stack(
            [torch.tensor(img_seq_len, device=DEVICE).long() for (_, img_seq_len) in batch])
        return (img_seqs, img_seq_len)

    def forward(net, x):
        (distrs, _) = x
        return net(distrs)

    def get_sample_strategy(self):
        s = self.config[STRATEGY]
        if s == SINGLE_SAMPLE_STRATEGY:
            input_mapping = [i for i in range(len(self.unstructured_dataset))]
            strat = strategy.SingleSampleStrategy(
                self.unstructured_dataset, input_mapping)
        elif s == SIMPLE_LIST_STRATEGY:
            strat = strategy.SimpleListStrategy(
                self.unstructured_dataset, self.im, self.config[MAX_LENGTH]
            )
        else:
            raise InvalidSampleStrategy(f"Sampling strategy {s} is invalid")
        return strat

    def get_preprocess_strategy(self):
        allowed = [PREPROCESS_IDENTITY]
        return self.preprocess_from_allowed_strategies(allowed)

    def generate_datapoint(self):
        return self.preprocess.preprocess(self.strategy.sample())

    def string_combine(inputs):
        return "".join(str(s) for s in inputs)

    def get_input_mapping(config):
        max_length = config[MAX_LENGTH]
        ud = get_unstructured_dataset_static(config)
        element_input_mapping = input.DiscreteInputMapping(
            ud.input_mapping(ud), id)
        if ud == unstructured_dataset.HWFDataset:
            combine = PaddedListDataset.string_combine
        else:
            raise Exception(f'invalid unstructured dataset: {ud}')
        return input.PaddedListInputMapping(max_length, element_input_mapping, combine)

    def distrs_to_input(distrs, x, config):
        lengths = [l.item() for l in x[1]]
        return input.PaddedListInput(distrs, lengths)


class StringDataset(StructuredDataset):
    def __init__(self, config, dataset_size, unstructured_dataset):
        super().__init__(config=config,
                         dataset_size=dataset_size,
                         unstructured_dataset=unstructured_dataset,
                         call_black_box_for_gt=True)

    def __len__(self):
        return self.dataset_size

    @staticmethod
    def collate_fn(batch, config):
        return torch.stack([torch.stack(item) for item in batch])

    def forward(net, x):
        batch_size, length, _, _, _ = x.shape
        return net(x.flatten(start_dim=0, end_dim=1)).view(batch_size, length, -1)

    def get_sample_strategy(self):
        length = self.config[LENGTH]
        s = self.config[STRATEGY]
        if s == SIMPLE_LIST_STRATEGY:
            strat = strategy.SimpleListStrategy(
                self.unstructured_dataset, self.im, length)
        else:
            raise InvalidSampleStrategy(f"Sampling strategy {s} is invalid")
        return strat

    def get_preprocess_strategy(self):
        allowed = [PREPROCESS_IDENTITY, PREPROCESS_SORT, PREPROCESS_PALINDROME]
        return self.preprocess_from_allowed_strategies(allowed)

    def generate_datapoint(self):
        samples = self.preprocess.preprocess(self.strategy.sample())
        imgs, string_list = zip(*samples)
        string = ''.join(str(self.im[n]) for n in string_list)
        return (imgs, string)

    def combine(inputs):
        return "".join(str(s) for s in inputs)

    def get_input_mapping(config):
        length = config[LENGTH]
        ud = get_unstructured_dataset_static(config)
        element_input_mapping = input.DiscreteInputMapping(
            ud.input_mapping(ud), StringDataset.combine)
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
    elif sd == SINGLE_INT_LIST_TYPE and MAX_LENGTH in config:
        return PaddedListDataset
    elif sd == SINGLE_INT_LIST_TYPE:
        return SingleIntListDataset
    elif sd == SINGLE_INT_LIST_LIST_TYPE:
        return SingleIntListListDataset
    elif sd == INT_LIST_TYPE:
        return IntListDataset
    elif sd == STRING_TYPE and MAX_LENGTH in config:
        return PaddedListDataset
    elif sd == STRING_TYPE:
        return StringDataset
    elif sd == STRING_LIST_TYPE:
        return StringListDataset
    else:
        raise UnknownStructuredDataset(f"Unknown dataset: {sd}")
