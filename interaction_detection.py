import numpy as np
from itertools import combinations
from random import sample
from game_simulation_utils import *
import pandas as pd
# Step 1: Generate a binary matrix
feature_no = 10
matrix = np.random.randint(0, 2, (500, feature_no))

results = []

for i in range(120):
    # Step 2: Generate subsets of size 2 to 5
    subset_sizes = range(2, 5)
    subsets = [sample(list(combinations(range(feature_no), size)), 2) for size in subset_sizes]
    subsets_flat = [s for sublist in subsets for s in sublist]  # Flatten the list

    matrix_extended = np.copy(matrix)
    # Step 3: Extend the matrix
    for subset in subsets_flat:
        product_col = np.prod(matrix_extended[:, subset], axis=1)
        matrix_extended = np.column_stack((matrix_extended, product_col))

    # Step 4: Create a vector and perform matrix-vector multiplication
    m = np.random.rand(matrix_extended.shape[1])
    v = np.dot(matrix_extended, m)


    shapley_value, alpha, interactions = gemfix_reg(matrix, v, np.ones_like(v))

    computed_subsets = [item[0] for item in interactions]
    # Step 6: Compute precision and recall
    for size, true_subsets in zip(subset_sizes, subsets):
        computed_for_size = [s for s in computed_subsets if len(s) == size]
        true_set = set(map(frozenset, true_subsets))
        computed_set = set(map(frozenset, computed_for_size))
        true_positives = len(true_set & computed_set)
        false_positive = len(computed_set) - true_positives
        false_negative = len(true_set) - true_positives
        results.append((i, size, true_positives, false_positive, false_negative))

size2 = [item for item in results if item[1] == 2]
size3 = [item for item in results if item[1] == 3]
size4 = [item for item in results if item[1] == 4]

df = pd.DataFrame(results, columns=['index', 'subsetsize', 'tp', 'fp', 'fn'])
df.to_excel('results/synthesized experiment/interactions.xlsx')

