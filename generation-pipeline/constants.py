
# Function
PY_PROGRAM = "py_func"

# Hyperparameters
LEARNING_RATE = "learning_rate"
BATCH_SIZE_TRAIN = "batch_size_train"
BATCH_SIZE_TEST = "batch_size_test"

DATASET_SIZE_TRAIN = "dataset_size_train"
DATASET_SIZE_TEST = "dataset_size_test"

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

# Datasets
DATASET = "dataset"
MNIST = "MNIST"
EMNIST = "EMNIST"
SVHN = "SVHN"
HWF_SYMBOL = "HWF"
MNIST_VIDEO = "MNIST_VIDEO"
MNIST_GRID = "MNIST_GRID"
MNIST_0TO4 = "MNIST_0TO4"
COFFEE_LEAF = "COFFEE_LEAF"

# Dataset generation strategies
STRATEGY = "strategy"
SINGLETON_STRATEGY = "singleton"
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
PREPROCESS_COFFEE = 'preprocess_coffee'

# Output
OUTPUT = "output"
OUTPUT_MAPPING = "output_mapping"
RANGE = "range"
OUTPUT_MAPPING_RANGE = "output_mapping_range"
START = "start"
END = "end"
UNKNOWN = "unknown"
FALLBACK = "fallback"

# Reserved failure
RESERVED_FAILURE = "__RESERVED_FAILURE__"
