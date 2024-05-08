import numpy as np

def sum_m(*inputs):
    return sum(inputs)

def add_sub(a, b, c):
  return a + b - c

def eq(a, b):
  return a==b

def how_many_3_4(a, b, c, d, e, f, g, h):
  digit_lst = [a, b, c, d, e, f, g, h]
  return sum((n == 3 or n == 4) for n in digit_lst)

def less_than(a, b):
    return a < b

def mod_2(a, b):
    return a % (b + 1)

def mult_2(a, b):
  return a * b

def not_3_4(a):
  return a!=3 and a!=4

def add_mod_3(a, b):
  return (a + b) % 3

def sort(*num_list):
  num_list = list(num_list)
  return np.argsort(num_list)