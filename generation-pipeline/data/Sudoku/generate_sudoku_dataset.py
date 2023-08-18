import random
from random import sample
import csv


def generate_board(base):
    side = base*base

    # pattern for a baseline valid solution

    def pattern(r, c): return (base*(r % base)+r//base+c) % side

    # randomize rows, columns and numbers (of valid base pattern)
    def shuffle(s): return sample(s, len(s))

    rBase = range(base)
    rows = [g*base + r for g in shuffle(rBase) for r in shuffle(rBase)]
    cols = [g*base + c for g in shuffle(rBase) for c in shuffle(rBase)]
    nums = shuffle(range(1, base*base+1))

    # produce board using randomized baseline pattern
    board = [[nums[pattern(r, c)] for c in cols] for r in rows]

    return board


def get_boards(base, name):
    length = base * base
    min_blanks = 6 if base == 2 else 16
    max_blanks = 12 if base == 2 else 52
    with open(name, 'w') as file:
        writer = csv.writer(file)
        for _ in range(100000):
            board = generate_board(base)
            cp_board = [[c for c in r] for r in board]
            n_blanks = random.randint(min_blanks, max_blanks)
            for _ in range(n_blanks):
                # Randomly remove a digit for each blank
                r_row = random.randint(0, length-1)
                r_col = random.randint(0, length-1)
                while not cp_board[r_row][r_col]:
                    r_row = random.randint(0, length-1)
                    r_col = random.randint(0, length-1)
                cp_board[r_row][r_col] = 0
            flattened_board = ''.join([str(c) for r in board for c in r])
            flattened_cp_board = ''.join([str(c) for r in cp_board for c in r])
            writer.writerow([flattened_cp_board, flattened_board])
        file.close()


get_boards(2, '4x4sudoku.csv')
get_boards(3, '9x9sudoku.csv')
