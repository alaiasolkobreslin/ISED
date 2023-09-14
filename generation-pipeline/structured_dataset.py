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
        elif s == PREPROCESS_SUDOKU_BOARD:
            strat = preprocess.PreprocessSudokuBoard()
        elif s == PREPROCESS_PALINDROME:
            strat = preprocess.PreprocessPalindrome()
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


class SudokuDataset(StructuredDataset):
    def __init__(self, config, dataset_size, unstructured_dataset):
        super().__init__(config=config,
                         dataset_size=dataset_size,
                         unstructured_dataset=unstructured_dataset,
                         call_black_box_for_gt=True)

    def __len__(self):
        return self.dataset_size

    @staticmethod
    def collate_fn(batch, config):
        batch = [(list(z), l, b) for (z, l, b) in batch]
        max_blanks = config[MAX_BLANKS]
        zero_img = torch.zeros_like(batch[0][0][0][0])

        def pad_zero(img_seq):
            img_seq = list(img_seq)
            return img_seq + [zero_img] * (max_blanks - len(img_seq))
        img_seqs = torch.stack([torch.stack(pad_zero(list(img_seq)[0]))
                               for (img_seq, _, _) in batch])
        img_seq_len = torch.stack(
            [torch.tensor(img_seq_len).long() for (_, img_seq_len, _) in batch])
        bool_boards = torch.stack([torch.from_numpy(bool_board)
                                  for (_, _, bool_board) in batch])

        return (img_seqs, img_seq_len, bool_boards)

    def forward(net, x):
        (distrs, _, _) = x
        batch_size, length, _, _, _ = distrs.shape
        return net(distrs.flatten(start_dim=0, end_dim=1)).view(batch_size, length, -1)

    def get_sample_strategy(self):
        s = self.config[STRATEGY]
        n_rows = self.config[N_ROWS]
        n_cols = self.config[N_COLS]
        if s == SUDOKU_PROBLEM_STRATEGY:
            strat = strategy.SudokuProblemStrategy(
                self.unstructured_dataset, n_rows, n_cols)
        elif s == SUDOKU_RANDOM_STRATEGY:
            strat = strategy.SudokuRandomStrategy(
                self.unstructured_dataset, n_rows, n_cols)
        else:
            raise InvalidSampleStrategy(f"Sampling strategy {s} is invalid")
        return strat

    def get_preprocess_strategy(self):
        allowed = [PREPROCESS_SUDOKU_BOARD]
        return self.preprocess_from_allowed_strategies(allowed)

    def generate_datapoint(self):
        samples, length, bool_board = self.strategy.sample()
        (samples, board) = self.preprocess.preprocess(samples, bool_board)
        s = [z for z in zip(*samples)]
        return ((s, length, bool_board), board)

    def combine(input):
        input, bool_board = input
        n_rows, n_cols = bool_board.shape
        result = []
        idx = 0
        for r in range(n_rows):
            current_row = []
            for c in range(n_cols):
                if bool_board[r][c]:
                    current_row.append(str(input[idx]))
                    idx += 1
                else:
                    current_row.append('.')
            result.append(current_row)
        return result

    def get_input_mapping(config):
        ud = get_unstructured_dataset_static(config)
        max_length = config[MAX_BLANKS]
        digit_input_mapping = input.DiscreteInputMapping(
            ud.input_mapping(ud), id)
        return input.PaddedListInputMappingSudoku(max_length, digit_input_mapping, SudokuDataset.combine)

    def distrs_to_input(distrs, x, config):
        lengths = [l.item() for l in x[1]]
        return input.PaddedListInputSudoku(distrs, lengths, x[2])


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


class VideoDataset(StructuredDataset):
    def __init__(self, config, dataset_size, unstructured_dataset):
        super().__init__(config=config,
                         dataset_size=dataset_size,
                         unstructured_dataset=unstructured_dataset,
                         call_black_box_for_gt=True)

    def __len__(self):
        return self.dataset_size

    @staticmethod
    def collate_fn(batch, _):
        return torch.stack(batch)

    def forward(net, x):
        return net(x)

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
        return samples

    def get_input_mapping(config):
        ud = get_unstructured_dataset_static(config)
        length = config[LENGTH]
        element_input_mapping = input.DiscreteInputMapping(
            ud.input_mapping(ud), id)
        return input.ListInputMapping(length, element_input_mapping, id)

    def distrs_to_input(distrs, x, config):
        length = config[LENGTH]
        return input.VideoInput(distrs[0], distrs[1], length)


class CoffeeLeafDataset(StructuredDataset):
    def __init__(self, config, dataset_size, unstructured_dataset):
        super().__init__(config=config,
                         dataset_size=dataset_size,
                         unstructured_dataset=unstructured_dataset,
                         call_black_box_for_gt=False)

    def __len__(self):
        return self.dataset_size

    @staticmethod
    def collate_fn(batch, config):
        max_len = config[MAX_LENGTH]
        zero_img = torch.zeros_like(batch[0][0][0])

        def pad_zero(img_seq): return img_seq + \
            [zero_img] * (max_len - len(img_seq))
        def pad_zero_areas(areas): return areas + \
            [0] * (max_len - len(areas))
        img_seqs = torch.stack([torch.stack(pad_zero(img_seq))
                               for (img_seq, _) in batch])
        area_seqs = torch.stack(
            [torch.Tensor(pad_zero_areas(areas)) for (_, areas) in batch])
        img_seq_len = torch.stack(
            [torch.tensor(len(areas)).long() for (_, areas) in batch])
        return (img_seqs, area_seqs, img_seq_len)

    def forward(net, x):
        (distrs, _, _) = x
        return net(distrs)

    def get_sample_strategy(self):
        s = self.config[STRATEGY]
        if s == SINGLE_SAMPLE_STRATEGY:
            # input_mapping = [i for i in range(len(self.unstructured_dataset))]
            input_mapping = [i for i in range(1, 6)]
            strat = strategy.SingleSampleStrategy(
                self.unstructured_dataset, input_mapping)
        else:
            raise InvalidSampleStrategy(f"Sampling strategy {s} is invalid")
        return strat

    def get_preprocess_strategy(self):
        allowed = [PREPROCESS_IDENTITY]
        return self.preprocess_from_allowed_strategies(allowed)

    def generate_datapoint(self):
        samples = self.preprocess.preprocess(self.strategy.sample())
        return samples

    def get_input_mapping(config):
        max_length = config[MAX_LENGTH]
        element_input_mapping = input.DiscreteInputMapping(
            [0, 1], id)
        return input.PaddedListInputMappingCoffee(max_length, element_input_mapping, id)

    def distrs_to_input(distrs, x, _):
        lengths = [l.item() for l in x[2]]
        areas = [[int(a.item()) for a in area] for area in x[1]]
        return input.CoffeeInput(distrs, lengths, areas)


class TokensDataset(StructuredDataset):
    def __init__(self, config, dataset_size, unstructured_dataset):
        super().__init__(config=config,
                         dataset_size=dataset_size,
                         unstructured_dataset=unstructured_dataset,
                         call_black_box_for_gt=True)

    def __len__(self):
        return self.dataset_size

    @staticmethod
    def collate_fn(batch, config):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        token_type_ids = torch.stack(
            [item['token_type_ids'] for item in batch])
        attention_mask = torch.stack(
            [item['attention_mask'] for item in batch])
        return (input_ids, token_type_ids, attention_mask)

    def forward(net, x):
        input_id, _, attention_mask = x
        input_id = torch.squeeze(input_id)
        attention_mask = torch.squeeze(attention_mask)
        return net(input_id=input_id, mask=attention_mask)

    def get_sample_strategy(self):
        s = self.config[STRATEGY]
        if s == SINGLE_SAMPLE_STRATEGY:
            strat = strategy.SingleSampleStrategy(
                self.unstructured_dataset, self.input_mapping)
        else:
            raise InvalidSampleStrategy(f"Sampling strategy {s} is invalid")
        return strat

    def get_preprocess_strategy(self):
        allowed = [PREPROCESS_IDENTITY]
        return self.preprocess_from_allowed_strategies(allowed)

    def generate_datapoint(self):
        samples = self.preprocess.preprocess(self.strategy.sample())
        return samples

    def get_input_mapping(config):
        max_length = config[MAX_LENGTH]
        element_input_mapping = input.DiscreteInputMapping(
            [i for i in range(9)], id)
        return input.ListInputMapping(length=max_length, element_input_mapping=element_input_mapping, combine=id)

    def distrs_to_input(distrs, x, config):
        _, length, _ = distrs.shape
        return input.ListInput(tensor=distrs, length=length)


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
    elif ud == MNIST_1TO4:
        return unstructured_dataset.MNISTDataset_1to4
    elif ud == MNIST_2TO9:
        return unstructured_dataset.MNISTDataset_2to9
    elif ud == COFFEE_LEAF_RUST:
        return unstructured_dataset.CoffeeLeafRustDataset
    elif ud == COFFEE_LEAF_MINER:
        return unstructured_dataset.CoffeeLeafMinerDataset
    elif ud == CONLL2003:
        return unstructured_dataset.CoNLL2003Dataset
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
    elif sd == SUDOKU_TYPE:
        return SudokuDataset
    elif sd == VIDEO_DIGIT_TYPE:
        return VideoDataset
    elif sd == LEAF_AREA_TYPE:
        return CoffeeLeafDataset
    elif sd == TOKEN_SEQUENCE_TYPE:
        return TokensDataset
    else:
        raise UnknownStructuredDataset(f"Unknown dataset: {sd}")
