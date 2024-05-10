
import numpy as np
from itertools import chain, combinations

import heapq
from math import ceil, log2
from itertools import chain, combinations

def try_add(queue, next_val, limit):
    """
    Tries to add a value to the priority queue if the queue isn't full or if it's smaller than the largest value.
    """
    if len(queue) == limit and queue[0] > next_val:
        heapq.heappop(queue)  # Remove the largest element to make space
    if len(queue) < limit:
        heapq.heappush(queue, next_val)

def get_lowest_abs_sum_comb(queue):
    """
    Returns all possible sums generated from the given queue (in Python, a list).
    """
    sum_comb = [0]
    for value in queue:
        sum_comb.extend([x + value for x in sum_comb])
    return sum_comb

def get_highest_sum_comb(queue, total, z):
    """
    Calculates the Z highest sums given the base total and a queue of lowest positive sums.
    """
    lowest_sums = get_lowest_abs_sum_comb(queue)
    lowest_sums.sort()
    
    if z == len(lowest_sums):
        result = lowest_sums
    else:
        result = lowest_sums[:z]

    return [total - x for x in result]

def get_max_sums(source, z):
    """
    Main function to calculate the Z highest subset sums using the Java logic.
    """
    total = 0
    limit = 31 - (len(bin(z)) - 2) + (z == 1 or bin(z).count('1') > 1)
    queue = []
    
    for next_val in source:
        abs_val = abs(next_val)
        try_add(queue, abs_val, limit)
        if next_val > 0:
            total += next_val

    return get_highest_sum_comb(queue, total, z)

# Example usage
input_array1 = [2, 4, 5]
input_array2 = [-2, 5, -3, 7, 9]
z1 = 3
z2 = 5

print(get_max_sums(input_array1, z1))  # Output should match the Java result
print(get_max_sums(input_array2, z2))  # Output should match the Java result


# Function to generate all subsets of a given list with size greater than one
def all_subsets(lst):
    return list(chain.from_iterable(combinations(lst, r) for r in range(2, len(lst)+1)))

# Create the binary matrix with 6 rows and 8 columns
# Each row contains at least 3 and at most 7 ones
num_rows = 6
num_cols = 8
binary_matrix = np.zeros((num_rows, num_cols), dtype=int)

# Populate each row with a random number of ones between 3 and 7
for i in range(num_rows):
    num_ones = np.random.randint(3, 8)
    ones_indices = np.random.choice(num_cols, num_ones, replace=False)
    binary_matrix[i, ones_indices] = 1

# Create a positive vector alpha of length 6
alpha = np.random.randint(1, 10, size=num_rows)

# Generate all possible subsets of the vector indices with length greater than one
all_alpha_indices = list(range(len(alpha)))
all_subsets_alpha = all_subsets(all_alpha_indices)

# Perform the multiplication and store nonzero indices
results = []
subset_values = []
for subset in all_subsets_alpha:
    subset_value = np.sum(alpha[list(subset)])
    subset_values.append([subset, subset_value])

    result_row = np.prod(binary_matrix[subset, :], axis=0)
    nonzero_indices = np.where(result_row != 0)[0]
    if len(nonzero_indices) > 1:
        results.append([ subset, nonzero_indices, subset_value ])


results.sort(key=lambda x: x[2], reverse=True)
subset_values.sort(key=lambda x: x[1], reverse=True)
## Get the same results in a more smart way 

smarter_results = []
all_sets = []

for set, value in subset_values:
    row_prod = np.prod(binary_matrix[list(set),:], axis=0)
    active_set = np.where(row_prod != 0)[0]
    all_sets.append([set, np.sum(alpha[list(set)])])

    if len(active_set) > 1:
        smarter_results.append([set, active_set, np.sum(alpha[list(set)])])


print('done!')