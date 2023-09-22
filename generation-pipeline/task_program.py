from heapq import merge
from typing import *
import torch
import pycosat
import numpy as np

# Scallop programs


def sum_2(digit_1, digit_2):
    return digit_1 + digit_2


def sum_3(digit_1, digit_2, digit_3):
    return digit_1 + digit_2 + digit_3


def sum_4(digit_1, digit_2, digit_3, digit_4):
    return digit_1 + digit_2 + digit_3 + digit_4


def sum_5(digit_1, digit_2, digit_3, digit_4, digit_5):
    return digit_1 + digit_2 + digit_3 + digit_4 + digit_5


def sum_6(digit_1, digit_2, digit_3, digit_4, digit_5, digit_6):
    return digit_1 + digit_2 + digit_3 + digit_4 + digit_5 + digit_6


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
    n = len(expr)
    for i in range(n):
        if i % 2 == 0 and not expr[i].isdigit():
            raise Exception("Invalid HWF")
        elif i % 2 == 1 and expr[i] not in ['+', '*', '-', '/']:
            raise Exception("Invalid HWF")
    return eval(expr)

# Leetcode problems


def add_two_numbers(number_1, number_2):
    # problem 2: https://leetcode.com/problems/add-two-numbers/
    return number_1 + number_2


def longest_substring_without_repeating_characters(s: str) -> str:
    # problem 3: https://leetcode.com/problems/longest-substring-without-repeating-characters/
    chars = {}
    start = 0
    max_length = 0

    for end, c in enumerate(s):
        if c in chars:
            start = max(start, chars[c] + 1)

        max_length = max(max_length, end - start + 1)
        chars[c] = end
    return max_length


def median_of_two_sorted_arrays(nums1, nums2):
    def kth(a, b, k):
        if not a:
            return b[k]
        if not b:
            return a[k]
        ia, ib = len(a) // 2, len(b) // 2
        ma, mb = a[ia], b[ib]

        # when k is bigger than the sum of a and b's median indices
        if ia + ib < k:
            # if a's median is bigger than b's, b's first half doesn't include k
            if ma > mb:
                return kth(a, b[ib + 1:], k - ib - 1)
            else:
                return kth(a[ia + 1:], b, k - ia - 1)
        # when k is smaller than the sum of a and b's indices
        else:
            # if a's median is bigger than b's, a's second half doesn't include k
            if ma > mb:
                return kth(a[:ia], b, k)
            else:
                return kth(a, b[:ib], k)
    l = len(nums1) + len(nums2)
    if l % 2 == 1:
        return kth(nums1, nums2, l // 2)
    else:
        return (kth(nums1, nums2, l // 2) + kth(nums1, nums2, l // 2 - 1)) / 2.


# def median_of_two_sorted_arrays(nums1, nums2):
#     # problem 4: https://leetcode.com/problems/median-of-two-sorted-arrays/
#     n1 = len(nums1)
#     n2 = len(nums2)

#     # If nums1 is larger than nums2, swap them to ensure n1 is smaller than n2.
#     if n1 > n2:
#         return median_of_two_sorted_arrays(nums2, nums1)

#     l = 0
#     r = n1
#     while l <= r:
#         mid1 = (l + r) / 2
#         mid2 = (n1 + n2 + 1) / 2 - mid1

#         maxLeft1 = nums1[mid1-1] if mid1 != 0 else float('-inf')
#         minRight1 = nums1[mid1] if mid1 != n1 else float('inf')

#         maxLeft2 = nums2[mid2-1] if mid2 != 0 else float('-inf')
#         minRight2 = nums2[mid2] if mid2 != n2 else float('inf')

#         if maxLeft1 <= minRight2 and maxLeft2 <= minRight1:
#             if (n1 + n2) % 2 == 0:
#                 return float(max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) / 2
#             else:
#                 return float(max(maxLeft1, maxLeft2))
#         elif maxLeft1 > minRight2:
#             r = mid1 - 1
#         else:
#             l = mid1 + 1

#     return -1


def longest_palindromic_substring(s: str) -> str:
    # problem 5: https://leetcode.com/problems/longest-palindromic-substring/
    def expand(string, a, b):
        while a >= 0 and b < len(string) and string[a] == string[b]:
            a -= 1
            b += 1
        return string[a+1:b]

    ans = ''
    for i in range(len(s)):
        ans = max(ans, expand(s, i, i), expand(s, i, i+1), key=len)
    return ans


def reverse_integer(x):
    # problem 7: https://leetcode.com/problems/reverse-integer/
    reversed = x[::-1]
    string = ''.join(str(d) for d in reversed)
    output = int(string)
    return output


def palindrome_number(x):
    # problem 9: https://leetcode.com/problems/palindrome-number/
    x = list(str(x))
    x_cmp = x[:]
    x_cmp.reverse()
    return x == x_cmp


def integer_to_roman(x):
    # problem 12: https://leetcode.com/problems/integer-to-roman/
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


def longest_common_prefix(strs):
    # problem 14: https://leetcode.com/problems/longest-common-prefix/
    if len(strs) == 0:
        return ""
    else:
        s1, s2 = max(strs), min(strs)
        i, match = 0, 0
        while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
            i, match = i+1, match + 1
        return s1[0:match]


def letter_combinations_of_a_phone_number(number):
    # problem 17: https://leetcode.com/problems/letter-combinations-of-a-phone-number/
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


def merge_two_sorted_lists(list1, list2):
    # problem 21: https://leetcode.com/problems/merge-two-sorted-lists/
    return list(merge(list1, list2))


def remove_duplicates_from_sorted_array(nums):
    # problem 26: https://leetcode.com/problems/remove-duplicates-from-sorted-array/
    nums = list(nums)
    count = 0
    for i in range(len(nums)):
        if i < len(nums) - 2 and nums[i] == nums[i + 1]:
            continue
        nums[count] = nums[i]
        count += 1
    return count


def valid_sudoku(board):
    # problem 36: https://leetcode.com/problems/valid-sudoku/
    """
    :type board: List[List[str]]
    :rtype: bool
    """
    for row in board:
        for item in row:
            if item not in ['.', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                return False
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


def sudoku_solver(board):
    # problem 37: https://leetcode.com/problems/sudoku-solver/
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


def trapping_rain_water(height):
    # problem 42: https://leetcode.com/problems/trapping-rain-water/
    n = len(height)
    left, right, left_max, right_max, water = 0, n - 1, 0, 0, 0
    while left <= right:
        if height[left] <= height[right]:
            if height[left] > left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] > right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    return water


def permutations_ii(nums):
    # problem 47: https://leetcode.com/problems/permutations-ii/
    nums = list(nums)
    n = len(nums)
    res = []

    def dfs(nums, l):
        if l == n-1:
            res.append(list(nums))
            return
        for i in set(nums[l:]):
            remaining = nums[l:]
            remaining.remove(i)
            dfs(nums[:l] + [i] + remaining, l+1)
    dfs(nums, 0)
    return res


def rotate_image(matrix):
    # problem 48: https://leetcode.com/problems/rotate-image/
    return list(zip(*matrix[::-1]))


def group_anagrams(strs):
    # problem 49: https://leetcode.com/problems/group-anagrams/
    d = {}
    for i in range(len(strs)):
        x = ''.join(sorted(strs[i]))
        if x not in d:
            d[x] = [strs[i]]
        else:
            d[x].append(strs[i])
    return list(d.values())


def maximum_subarray(nums):
    # problem 53: https://leetcode.com/problems/maximum-subarray/
    if len(nums) == 1:
        return nums[0]

    sum_result = [0 for i in range(len(nums))]
    sum_result[0] = nums[0]
    for i in range(1, len(nums)):
        if sum_result[i-1] < 0:
            sum_result[i] = nums[i]
        else:
            sum_result[i] = sum_result[i-1] + nums[i]
    return max(sum_result)


def spiral_matrix(matrix):
    # problem 54: https://leetcode.com/problems/spiral-matrix/
    matrix = list(matrix)
    res = []
    while matrix:
        res.extend(matrix.pop(0))
        matrix = [*zip(*matrix)][::-1]
    return res


def jump_game(nums):
    # problem 55: https://leetcode.com/problems/jump-game/
    i = 0
    reach = 0
    while i < len(nums) and i <= reach:
        reach = max(reach, i + nums[i])
        i += 1
    return i == len(nums)


def spiral_matrix_ii(n):
    # problem 59: https://leetcode.com/problems/spiral-matrix-ii/
    A, lo = [], n*n+1
    while lo > 1:
        lo, hi = lo - len(A), lo
        A = [range(lo, hi)] + list(zip(*A[::-1]))
    return A


def minimum_path_sum(grid):
    # problem 64: https://leetcode.com/problems/minimum-path-sum/
    M, N = len(grid), len(grid[0])
    for i in range(M):
        grid[i][0] = grid[i][0] + grid[i-1][0] if i > 0 else grid[i][0]
        for j in range(1, N):
            grid[i][j] = min(grid[i-1][j], grid[i][j-1]) + \
                grid[i][j] if i > 0 else grid[i][j-1]+grid[i][j]
    return grid[-1][-1]


def plus_one(digits):
    # problem 66: https://leetcode.com/problems/plus-one/
    digits = list(digits)
    length = len(digits) - 1
    while digits[length] == 9:
        digits[length] = 0
        length -= 1
    if(length < 0):
        digits = [1] + digits
    else:
        digits[length] += 1
    return digits


def climbing_stairs(n):
    # problem 70: https://leetcode.com/problems/climbing-stairs/
    if n == 1 or n == 2:
        return n
    prevPrev = 1
    prev = 2
    current = 0
    for _ in range(3, n+1):
        current = prevPrev + prev
        prevPrev = prev
        prev = current
    return current


def edit_distance(word1, word2):
    # problem 72: https://leetcode.com/problems/edit-distance/
    m = len(word1)
    n = len(word2)
    table = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        table[i][0] = i
    for j in range(n + 1):
        table[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                table[i][j] = table[i - 1][j - 1]
            else:
                table[i][j] = 1 + min(table[i - 1][j],
                                      table[i][j - 1], table[i - 1][j - 1])
    return table[-1][-1]


def combinations(n, k):
    # problem 77: https://leetcode.com/problems/combinations/
    ans = []

    def dfs(s: int, path: List[int]) -> None:
        if len(path) == k:
            ans.append(path.copy())
            return
        for i in range(s, n + 1):
            path.append(i)
            dfs(i + 1, path)
            path.pop()
    dfs(1, [])
    return ans


def largest_rectangle_in_histogram(heights):
    # problem 84: https://leetcode.com/problems/largest-rectangle-in-histogram/
    max_area, st = 0, []
    for idx, x in enumerate(heights):
        if len(st) == 0:
            st.append(idx)
        elif x >= heights[st[-1]]:
            st.append(idx)
        else:
            while st and heights[st[-1]] > x:
                min_height = heights[st.pop()]
                max_area = max(max_area, min_height*(idx-1 -
                               st[-1])) if st else max(max_area, min_height*idx)
            st.append(idx)
    while st:
        min_height = heights[st.pop()]
        max_area = max(max_area, min_height*(len(heights)-1 -
                       st[-1])) if st else max(max_area, min_height*len(heights))
    return max_area


def subsets_ii(nums):
    # problem 90: https://leetcode.com/problems/subsets-ii/
    res = [[]]
    nums = list(nums)
    nums.sort()
    for i in range(len(nums)):
        if i == 0 or nums[i] != nums[i - 1]:
            l = len(res)
        for j in range(len(res) - l, len(res)):
            res.append(res[j] + [nums[i]])
    return res


def decode_ways(s):
    # problem 91: https://leetcode.com/problems/decode-ways/
    if not s:
        return 0
    dp = [0 for _ in range(len(s) + 1)]
    # base case initialization
    dp[0] = 1
    dp[1] = 0 if s[0] == "0" else 1  # (1)
    for i in range(2, len(s) + 1):
        if 0 < int(s[i-1:i]) <= 9:  # (2)
            dp[i] += dp[i - 1]
        if 10 <= int(s[i-2:i]) <= 26:  # (3)
            dp[i] += dp[i - 2]
    return dp[len(s)]


def interleaving_string(s1, s2, s3):
    # problem 97: https://leetcode.com/problems/interleaving-string/
    r, c, l = len(s1), len(s2), len(s3)
    if r+c != l:
        return False
    queue, visited = [(0, 0)], set((0, 0))
    while queue:
        x, y = queue.pop(0)
        if x+y == l:
            return True
        if x+1 <= r and s1[x] == s3[x+y] and (x+1, y) not in visited:
            queue.append((x+1, y))
            visited.add((x+1, y))
        if y+1 <= c and s2[y] == s3[x+y] and (x, y+1) not in visited:
            queue.append((x, y+1))
            visited.add((x, y+1))
    return False


def best_time_to_buy_and_sell_stock(prices):
    # problem 121: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
    left = 0
    right = 1
    max_profit = 0
    while right < len(prices):
        currentProfit = prices[right] - prices[left]  # our current Profit
        if prices[left] < prices[right]:
            max_profit = max(currentProfit, max_profit)
        else:
            left = right
        right += 1
    return max_profit


def longest_consecutive_sequence(nums):
    # problem 128: https://leetcode.com/problems/longest-consecutive-sequence/
    nums = list(nums)
    nums.sort()
    longest, cur_longest = 0, min(1, len(nums))
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1]:
            continue
        if nums[i] == nums[i - 1] + 1:
            cur_longest += 1
        else:
            longest, cur_longest = max(longest, cur_longest), 1
    return max(longest, cur_longest)

# Other programs


def sort_list(x):
    return sorted(x)


def sort_integer_list(x):
    return sorted(x)


def char_identity(x):
    return x


def grid_identity(x):
    return x


def mnist_video_digits(x):
    digits, changes = x
    indices_with_changes = changes.nonzero()
    digits_with_changes = [digits[i[0]].nonzero().item()
                           for i in indices_with_changes]
    return digits_with_changes


def palindrome_string(str):
    return str == str[::-1]


def valid_mini_sudoku(board):
    """
    :type board: List[List[str]]
    :rtype: bool
    """
    # Check rows
    for i in range(4):
        d = {}
        for j in range(4):
            if board[i][j] == '.':
                pass
            elif board[i][j] in d:
                return False
            else:
                d[board[i][j]] = True
    # Check columns
    for j in range(4):
        d = {}
        for i in range(4):
            if board[i][j] == '.':
                pass
            elif board[i][j] in d:
                return False
            else:
                d[board[i][j]] = True
    # Check sub-boxes
    for m in range(0, 4, 2):
        for n in range(0, 4, 2):
            d = {}
            for i in range(n, n + 2):
                for j in range(m, m + 2):
                    if board[i][j] == '.':
                        pass
                    elif board[i][j] in d:
                        return False
                    else:
                        d[board[i][j]] = True
    return True


def solve_sudoku(length, root_length, boxes, board):

    def lit(i, j, d):
        return (length ** 2) * (i - 1) + length * (j - 1) + d

    clauses = []
    for i in range(1, length + 1):
        for j in range(1, length + 1):
            n = board[i-1][j-1]
            if n in [str(i) for i in range(1, length + 1)]:
                # add known values to clauses
                clauses.append([lit(i, j, int(n))])

    for i in range(1, length + 1):
        for j in range(1, length + 1):
            # each cell contains a value
            clauses.append([lit(i, j, d) for d in range(1, length + 1)])
            for d in range(1, length + 1):
                for n in range(d + 1, length + 1):
                    # no two values occupy the same cell
                    clauses.append([-lit(i, j, d), -lit(i, j, n)])

    def valid(cells):
        for i, xi in enumerate(cells):
            for j, xj in enumerate(cells):
                if i < j:
                    for d in range(1, length + 1):
                        clauses.append(
                            [-lit(xi[0], xi[1], d), -lit(xj[0], xj[1], d)])

    for i in range(1, length + 1):
        # check that rows contain no conflicts
        valid([(i, j) for j in range(1, length + 1)])
        # chceck that columns contain no conflicts
        valid([(j, i) for j in range(1, length + 1)])

    # check that boxes contain no conflicts
    for i in boxes:
        for j in boxes:
            valid([(i + k % root_length, j + k // root_length)
                  for k in range(length)])

    solved = pycosat.solve(clauses)
    if solved == 'UNSAT':
        return board
    sol = set(solved)

    def read_cell(i, j):
        # return the digit of cell i, j according to the solution
        for d in range(1, length + 1):
            if lit(i, j, d) in sol:
                return d

    solution = []
    for i in range(1, length + 1):
        solution_row = []
        for j in range(1, length + 1):
            solution_row.append(str(read_cell(i, j)))
        solution.append(solution_row)

    return solution


def mini_sudoku_solver(board):
    length = 4
    root_length = 2
    boxes = (1, 3)
    return solve_sudoku(length, root_length, boxes, board)


def sudoku_solver(board):
    length = 9
    root_length = 3
    boxes = (1, 4, 7)
    return solve_sudoku(length, root_length, boxes, board)


def sum_grid(grid):
    return sum(elt for row in grid for elt in row)


def sort_list_indices(x):
    return np.argsort(x, kind='mergesort').tolist()


def select_top_4(x):
    return np.argsort(x, kind='mergesort').tolist()[:4]


def rust_coffee_leaf_severity(selected_areas):
    quantiles = {
        "Q1": 37526.200000000004,
        "Q2": 63953.00000000001,
        "Q3": 100673.2,
        "Q4": 164972.80000000002
    }
    total_area = 0
    for (selected, area) in selected_areas:
        if selected:
            total_area += area
    if total_area < quantiles['Q1']:
        return 1
    elif total_area < quantiles['Q2']:
        return 2
    elif total_area < quantiles['Q3']:
        return 3
    elif total_area < quantiles['Q4']:
        return 4
    else:
        return 5


def miner_coffee_leaf_severity(selected_areas):
    quantiles = {
        "Q1": 354227.0,
        "Q2": 498202.4,
        "Q3": 620841.6,
        "Q4": 812283.6
    }
    total_area = 0
    for (selected, area) in selected_areas:
        if selected:
            total_area += area
    if total_area < quantiles['Q1']:
        return 1
    elif total_area < quantiles['Q2']:
        return 2
    elif total_area < quantiles['Q3']:
        return 3
    elif total_area < quantiles['Q4']:
        return 4
    else:
        return 5


def bio_tagging(tags):
    # {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    last_per_b = 0
    last_org_b = 0
    last_loc_b = 0
    last_misc_b = 0
    last_per_i = 0
    last_org_i = 0
    last_loc_i = 0
    last_misc_i = 0
    for i, tag in enumerate(tags):
        if tag == 1:  # B-PER
            last_per_b = i
        elif tag == 3:  # B-ORG
            last_org_b = i
        elif tag == 5:  # B-LOC
            last_loc_b = i
        elif tag == 7:  # B-MISC
            last_misc_b = i
        elif tag == 2:  # I-PER
            if last_per_b != i - 1 and last_per_i != i - 1:
                return False
            last_per_i = i
        elif tag == 4:  # I-ORG
            if last_org_b != i - 1 and last_org_i != i - 1:
                return False
            last_org_i = i
        elif tag == 6:  # I-LOC
            if last_loc_b != i - 1 and last_loc_i != i - 1:
                return False
            last_loc_i = i
        elif tag == 8:  # I-MISC
            if last_misc_b != i - 1 and last_misc_i != i - 1:
                return False
            last_misc_i = i
        else:
            continue
    return True


def ner_identity(x):
    return x


def reverse_string(str):
    return str[::-1]


dispatcher = {
    'sum_2': sum_2,
    'sum_3': sum_3,
    'sum_4': sum_4,
    'sum_5': sum_5,
    'sum_6': sum_6,
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
    'longest_substring_without_repeating_characters': longest_substring_without_repeating_characters,
    'median_of_two_sorted_arrays': median_of_two_sorted_arrays,
    'longest_palindromic_substring': longest_palindromic_substring,
    'reverse_integer': reverse_integer,
    'palindrome_number': palindrome_number,
    'integer_to_roman': integer_to_roman,
    'letter_combinations_of_a_phone_number': letter_combinations_of_a_phone_number,
    'merge_two_sorted_lists': merge_two_sorted_lists,
    'remove_duplicates_from_sorted_array': remove_duplicates_from_sorted_array,
    'valid_sudoku': valid_sudoku,
    'sudoku_solver': sudoku_solver,
    'trapping_rain_water': trapping_rain_water,
    'permutations_ii': permutations_ii,
    'rotate_image': rotate_image,
    'group_anagrams': group_anagrams,
    'maximum_subarray': maximum_subarray,
    'spiral_matrix': spiral_matrix,
    'jump_game': jump_game,
    'spiral_matrix_ii': spiral_matrix_ii,
    'minimum_path_sum': minimum_path_sum,
    'plus_one': plus_one,
    'climbing_stairs': climbing_stairs,
    'edit_distance': edit_distance,
    'combinations': combinations,
    'largest_rectangle_in_histogram': largest_rectangle_in_histogram,
    'subsets_ii': subsets_ii,
    'decode_ways': decode_ways,
    'interleaving_string': interleaving_string,
    'best_time_to_buy_and_sell_stock': best_time_to_buy_and_sell_stock,
    'longest_consecutive_sequence': longest_consecutive_sequence,

    'sort_list': sort_list,
    'sort_integer_list': sort_integer_list,
    'char_identity': char_identity,
    'grid_identity': grid_identity,
    'mnist_video_digits': mnist_video_digits,
    'palindrome_string': palindrome_string,
    'valid_mini_sudoku': valid_mini_sudoku,
    'mini_sudoku_solver': mini_sudoku_solver,
    'sum_grid': sum_grid,
    'sort_list_indices': sort_list_indices,
    'select_top_4': select_top_4,
    'rust_coffee_leaf_severity': rust_coffee_leaf_severity,
    'miner_coffee_leaf_severity': miner_coffee_leaf_severity,
    'bio_tagging': bio_tagging,
    'ner_identity': ner_identity,
    'reverse_string': reverse_string,
}


def dispatch(name, dispatch_args):
    """
    Returns the result of calling function `name` with arguments `dispatch_args`
    """
    return dispatcher[name](*dispatch_args.values())
