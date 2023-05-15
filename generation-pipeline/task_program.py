import torch
from heapq import merge

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
    if type(x) is torch.Tensor:
        x = x.item()
    y = str(x)
    y = y.strip()
    y = y[::-1]
    output = int(y)
    if output >= 2 ** 31 - 1 or output <= -2 ** 31:
        return 0
    return output


# https://leetcode.com/problems/palindrome-number/

def palindrome_number(x):
    if type(x) is torch.Tensor:
        x = x.item()
    x = list(str(x))
    x_cmp = x[:]
    x_cmp.reverse()
    return x == x_cmp

# https://leetcode.com/problems/integer-to-roman/


def integer_to_roman(x):
    if type(x) is torch.Tensor:
        x = x.item()
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

# https://leetcode.com/problems/merge-two-sorted-lists/


def merge_two_sorted_lists(list1, list2):
    return list(merge(list1, list2))

# https://leetcode.com/problems/letter-combinations-of-a-phone-number/


def letter_combinations_of_a_phone_number(number):
    L = {'2': "abc", '3': "def", '4': "ghi", '5': "jkl",
         '6': "mno", '7': "pqrs", '8': "tuv", '9': "wxyz"}
    lenD, ans = len(number), []
    if number == "":
        return []

    def bfs(pos: int, st: str):
        if pos == lenD:
            ans.append(st)
        else:
            letters = L[number[pos]]
            for letter in letters:
                bfs(pos+1, st+letter)
    bfs(0, "")
    return ans

# https://leetcode.com/problems/valid-sudoku/


def valid_sudoku(board):
    """
    :type board: List[List[str]]
    :rtype: bool
    """
    # Check rows
    for i in range(9):
        d = {}
        for j in range(9):
            if board[i][j] == '.':
                pass
            elif board[i][j] in d:
                return False
            else:
                d[board[i][j]] = True
    # Check columns
    for j in range(9):
        d = {}
        for i in range(9):
            if board[i][j] == '.':
                pass
            elif board[i][j] in d:
                return False
            else:
                d[board[i][j]] = True
    # Check sub-boxes
    for m in range(0, 9, 3):
        for n in range(0, 9, 3):
            d = {}
            for i in range(n, n + 3):
                for j in range(m, m + 3):
                    if board[i][j] == '.':
                        pass
                    elif board[i][j] in d:
                        return False
                    else:
                        d[board[i][j]] = True
    return True

# https://leetcode.com/problems/sudoku-solver/


def sudoku_solver(board):
    def isValid(row: int, col: int, c: chr) -> bool:
        for i in range(9):
            if board[i][col] == c or \
               board[row][i] == c or \
               board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] == c:
                return False
        return True

    def solve(s: int) -> bool:
        if s == 81:
            return True

        i = s // 9
        j = s % 9

        if board[i][j] != '.':
            return solve(s + 1)

        for c in '123456789':
            if isValid(i, j, c):
                board[i][j] = c
                if solve(s + 1):
                    return True
                board[i][j] = '.'

        return False

    solve(0)

# https://leetcode.com/problems/longest-palindromic-substring/


def longest_palindromic_substring(s: str) -> str:
    def expand(string, a, b):
        while a >= 0 and b < len(string) and string[a] == string[b]:
            a -= 1
            b += 1
        return string[a+1:b]

    ans = ''
    for i in range(len(s)):
        ans = max(ans, expand(s, i, i), expand(s, i, i+1), key=len)
    return ans

# Other programs


def sort_list(x):
    return sorted(x)


def char_identity(x):
    return x


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
    'merge_two_sorted_lists': merge_two_sorted_lists,
    'letter_combinations_of_a_phone_number': letter_combinations_of_a_phone_number,
    'valid_sudoku': valid_sudoku,
    'sudoku_solver': sudoku_solver,
    'longest_palindromic_substring': longest_palindromic_substring,

    'sort_list': sort_list,
    'char_identity': char_identity,
}


def dispatch(name, dispatch_args):
    """
    Returns the result of calling function `name` with arguments `dispatch_args`
    """
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
