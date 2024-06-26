import string
import torch

# Function
PY_PROGRAM = "py_func"

# Hyperparameters
LEARNING_RATE = "learning_rate"
BATCH_SIZE_TRAIN = "batch_size_train"
BATCH_SIZE_TEST = "batch_size_test"

DATASET_SIZE_TRAIN = "dataset_size_train"
DATASET_SIZE_TEST = "dataset_size_test"

NO_DATASET_GENERATION = "no_dataset_generation"

# Inputs
INPUTS = "inputs"

# Input parameters
TYPE = "type"
NAME = "name"
N_DIGITS = "num_digits"
LENGTH = "length"
STR_LENGTH = "str_length"
MAX_LENGTH = "max_length"
N_ROWS = 'n_rows'
N_COLS = 'n_cols'
MAX_BLANKS = 'max_blanks'

# Types
DIGIT_TYPE = "digit"
INT_TYPE = "int"
CHAR_TYPE = "char"
SINGLE_INT_LIST_TYPE = "List[digit]"
INT_LIST_TYPE = "List[int]"
SINGLE_INT_LIST_LIST_TYPE = "List[List[digit]]"
INT_LIST_LIST_TYPE = "List[List[int]]"
STRING_TYPE = "str"
STRING_LIST_TYPE = "List[str]"
SUDOKU_TYPE = "Sudoku"
VIDEO_DIGIT_TYPE = "Video[digit]"
LEAF_AREA_TYPE = "leaf_area"
TOKEN_SEQUENCE_TYPE = "token_sequence"

# Datasets
DATASET = "dataset"
MNIST = "MNIST"
EMNIST = "EMNIST"
SVHN = "SVHN"
HWF_SYMBOL = "HWF"

# Dataset generation strategies
STRATEGY = "strategy"
SINGLE_SAMPLE_STRATEGY = "single_sample"
SIMPLE_LIST_STRATEGY = "list"
LIST_2D = "2d_list"

# Preprocessing
PREPROCESS = "preprocess"
PREPROCESS_IDENTITY = "id"
PREPROCESS_SORT = "sort"
PREPROCESS_PALINDROME = 'palindrome'

# Output
OUTPUT = "output"
OUTPUT_MAPPING = "output_mapping"
RANGE = "range"
OUTPUT_MAPPING_RANGE = "output_mapping_range"
START = "start"
END = "end"
UNKNOWN = "unknown"
FALLBACK = "fallback"
LIST_OUTPUT_MAPPING = "list_output_mapping"
INT_OUTPUT_MAPPING = 'int_output_mapping'
DISCRETE_OUTPUT_MAPPING = "discrete_output_mapping"
N_CLASSES = "n_classes"

# Reserved failure
RESERVED_FAILURE = "__RESERVED_FAILURE__"

# EMNIST mapping
digits_im = [str(i) for i in range(10)]
uppercase_im = list(string.ascii_uppercase)
lowercase_im = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
EMNIST_MAPPING = digits_im + uppercase_im + lowercase_im

# Loss aggregation
LOSS_AGGREGATOR = "loss_aggregator"
MIN_MAX = "min_max"
ADD_MULT = "add_mult"

# Device
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"