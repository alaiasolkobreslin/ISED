from heapq import merge
from typing import *
import torch
import numpy as np

# Scallop programs


def sum_2(digit_1, digit_2):
    return digit_1 + digit_2


def sum_3(digit_1, digit_2, digit_3):
    return digit_1 + digit_2 + digit_3


def sum_4(digit_1, digit_2, digit_3, digit_4):
    return digit_1 + digit_2 + digit_3 + digit_4

 
def add_mod_3(digit_1, digit_2):
    return (digit_1 + digit_2) % 3


def add_sub(digit_1, digit_2, digit_3):
    return digit_1 + digit_2 - digit_3


def eq_2(digit_1, digit_2):
    return digit_1 == digit_2


def how_many_3_or_4(x):
    return sum((n == 3 or n == 4) for n in x)


def how_many_3(x):
    return sum((n == 3) for n in x)


def how_many_not_3_and_not_4(x):
    return sum((n != 3 and n != 4) for n in x)


def how_many_not_3(x):
    return sum((n != 3) for n in x)


def identity(x):
    return x


def is_3_and_4(digit_1, digit_2):
    return (digit_1 == 3) and (digit_2 == 4)


def not_3_or_4(digit_1):
    return digit_1 != 3 and digit_1 != 4


def less_than(digit_1, digit_2):
    return digit_1 < digit_2


def mod_2(digit_1, digit_2):
    return digit_1 % (digit_2 + 1)


def mult_2(digit_1, digit_2):
    return digit_1 * digit_2


dispatcher = {
    'sum_2': sum_2,
    'sum_3': sum_3,
    'sum_4': sum_4,
    'add_mod_3': add_mod_3,
    'add_sub': add_sub,
    'eq_2': eq_2,
    'how_many_3_or_4': how_many_3_or_4,
    'how_many_3': how_many_3,
    'how_many_not_3_and_not_4': how_many_not_3_and_not_4,
    'how_many_not_3': how_many_not_3,
    'identity': identity,
    'is_3_and_4': is_3_and_4,
    'not_3_or_4': not_3_or_4,
    'less_than': less_than,
    'mod_2': mod_2,
    'mult_2': mult_2
}


def dispatch(name, dispatch_args):
    """
    Returns the result of calling function `name` with arguments `dispatch_args`
    """
    return dispatcher[name](*dispatch_args.values())
