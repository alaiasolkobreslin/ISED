
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
MNIST_VIDEO = "MNIST_VIDEO"
MNIST_1TO4 = "MNIST_1TO4"
COFFEE_LEAF_MINER = "COFFEE_LEAF_MINER"
COFFEE_LEAF_RUST = "COFFEE_LEAF_RUST"
CONLL2003 = "CoNLL2003"

# Dataset generation strategies
STRATEGY = "strategy"
SINGLE_SAMPLE_STRATEGY = "single_sample"
SIMPLE_LIST_STRATEGY = "list"
LIST_2D = "2d_list"
SUDOKU_PROBLEM_STRATEGY = "sudoku_probem_sample"
SUDOKU_RANDOM_STRATEGY = "sudoku_random_sample"

# Preprocessing
PREPROCESS = "preprocess"
PREPROCESS_IDENTITY = "id"
PREPROCESS_SORT = "sort"
PREPROCESS_SUDOKU_BOARD = 'make_board'
PREPROCESS_PALINDROME = 'palindrome'

# Output
OUTPUT = "output"
OUTPUT_MAPPING = "output_mapping"
SUDOKU_OUTPUT_MAPPING = "sudoku_output_mapping"
RANGE = "range"
OUTPUT_MAPPING_RANGE = "output_mapping_range"
START = "start"
END = "end"
UNKNOWN = "unknown"
FALLBACK = "fallback"
SUDOKU_OUTPUT_MAPPING = "sudoku_output_mapping"
LIST_OUTPUT_MAPPING = "list_output_mapping"
INT_OUTPUT_MAPPING = 'int_output_mapping'

# Reserved failure
RESERVED_FAILURE = "__RESERVED_FAILURE__"
