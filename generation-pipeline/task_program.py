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


def not_3_or_4(digit_1, digit_2):
    return (digit_1 != 3) and (digit_2 != 4)


def less_than(digit_1, digit_2):
    return digit_1 < digit_2


def mod_2(digit_1, digit_2):
    return digit_1 % (digit_2 + 1)


def mult_2(digit_1, digit_2):
    return digit_1 * digit_2


def hwf(expr):
    try:
        return eval(expr)
    except Exception:
        return None

# Leetcode problems

# https://leetcode.com/problems/add-two-numbers/


def add_two_numbers(number_1, number_2):
    return number_1 + number_2

# https://leetcode.com/problems/reverse-integer/


def reverse_integer(x):
    y = str(x)
    y = y.strip()
    y = y[::-1]
    output = int(y)
    if output >= 2 ** 31 - 1 or output <= -2 ** 31:
        return 0
    return output


# https://leetcode.com/problems/palindrome-number/

def palindrome_number(x):
    x_cmp = x[:]
    x_cmp.reverse()
    return x == x_cmp

# https://leetcode.com/problems/integer-to-roman/


def integer_to_roman(x):
    rmap = {
        1: "I",
        4: "IV",
        5: "V",
        9: "IX",
        10: "X",
        40: "XL",
        50: "L",
        90: "XC",
        100: "C",
        400: "CD",
        500: "D",
        900: "CM",
        1000: "M"
    }
    seq = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    so_far, idx = [], 0
    while x > 0:
        if x >= seq[idx]:
            so_far.append(rmap[seq[idx]])
            x -= seq[idx]
        else:
            idx += 1
        return "".join(so_far)


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
    'mult_2': mult_2,
    'hwf': hwf,

    'add_two_numbers': add_two_numbers,
    'reverse_integer': reverse_integer,
    'palindrome_number': palindrome_number,
    'integer_to_roman': integer_to_roman,
}


def dispatch(name, dispatch_args):
    args = '('
    for i, k in enumerate(dispatch_args):
        arg = dispatch_args[k]
        if type(arg) is str:
            arg = "\'" + dispatch_args[k] + "\'"
        next_str = k + '=' + str(arg)
        if i != len(dispatch_args) - 1:
            next_str += ', '
        args += next_str
    args += ')'
    call = name + args
    return eval(call, {'__builtins__': None}, dispatcher)
