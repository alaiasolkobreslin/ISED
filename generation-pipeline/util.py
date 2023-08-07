from constants import *
from typing import *


def get_hashable_elem(elem: Any):
    if type(elem) is list or type(elem) is tuple:
        return tuple(get_hashable_elem(e) for e in elem)
    else:
        return elem
