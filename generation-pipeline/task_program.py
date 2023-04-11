# Scallop programs
def sum_2(digit_1, digit_2):
    return digit_1 + digit_2


def sum_3(digit_1, digit_2, digit_3):
    return digit_1 + digit_2 + digit_3


def sum_4(digit_1, digit_2, digit_3, digit_4):
    return digit_1 + digit_2 + digit_3 + digit_4


def hwf(symbols):
    pass

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
    'hwf': hwf,

    'add_two_numbers': add_two_numbers,
    'reverse_integer': reverse_integer,
    'palindrome_number': palindrome_number,
    'integer_to_roman': integer_to_roman,
}


def dispatch(name, dispatch_args):
    args = '('
    for i, k in enumerate(dispatch_args):
        next_str = k + '=' + str(dispatch_args[k])
        if i != len(dispatch_args) - 1:
            next_str += ', '
        args += next_str
    args += ')'
    call = name + args
    return eval(call, {'__builtins__': None}, dispatcher)
