
def sum_2(digit_1, digit_2):
    return digit_1 + digit_2

def add_two_numbers(number_1, number_2):
    return number_1 + number_2

dispatcher = {
    'sum_2' : sum_2,
    'add_two_numbers' : add_two_numbers,
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
    return eval(call, {'__builtins__' : None} , dispatcher)
