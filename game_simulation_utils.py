from itertools import combinations, chain
from random import seed
import random
from math import comb , factorial

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.svm import SVR, LinearSVR
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, check_array

import numpy as np
from scipy.linalg import cholesky, cho_solve

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import lars_path, LassoLarsIC
import heapq

''' 
Game generation and subset generation/selection
'''

player_no_exact = 12
num_samples = 100000

# Generate a random game
def generate_game(N, q=None, essential=True):
    if q == None: q = N
    if ~essential: q = 1
    players = list(range(1, N + 1))

    subsets_all = all_subsets(players)
    if N <= player_no_exact:
        subset_values_mobius = {subset: np.random.uniform(-1, 1) / len(subset) if len(subset) <= q and len(subset) > 0 else 0 for subset in subsets_all }
        subset_values_mobius[frozenset({})] = 0
    else:
        sample_size = np.min((2**N, num_samples))
        subsets_all = random_subsets(players, sample_size)
        subset_values_mobius = {subset: np.random.uniform(-1, 1) / len(subset) if len(subset) <= q and len(subset) > 0 else 0 for subset in subsets_all }
        subset_values_mobius[frozenset({})] = 0
        
    subset_values = {}
    for set in subsets_all:
        subset_values[set] = sum([subset_values_mobius[x] for x in subset_values_mobius if x.issubset(set)])

    return players, subset_values_mobius, subsets_all, subset_values

# Function to generate random subsets
def random_subsets(full_set, num_samples=1000):
    """Generate a list of random subsets of the full set."""
    subsets = []
    for _ in range(num_samples):
        k = np.random.randint(0, len(full_set))
        subsets.append(frozenset(np.random.choice(full_set, k, replace=False)))
    return subsets

# Build all subsets
def all_subsets(full_set):
    """Generate all subsets of the full set."""
    subsets = []
    for r in range(len(full_set) + 1):
        for subset in combinations(full_set, r):
            subsets.append(frozenset(subset))
    return subsets

'''
Exact Shapley value Calculation 
'''

# Calculating exact Shpaley Value based on the generated game
def exact_Shapley_value(subset_values_mobius, N):
    shapley_values = np.zeros((N,))
    for i in range(N):
        shapley_values[i] = sum([subset_values_mobius[x] / len(x) for x in subset_values_mobius if frozenset({i+1}).issubset(x)])
    return shapley_values    


def interactions_from_alpha(matrix, values):
        interactions_dict = {}
        
        # Process each row in the matrix along with its corresponding value
        for row, value in zip(matrix, values):
            #if abs(value) < np.mean(np.abs(values)) and abs(value) < 1e-5: continue 
            if abs(value) < 1e-5: continue 

            n = len(row)
            indices = [i for i in range(n) if row[i] == 1]                
            
            
            # Generate all subsets for the indices where the row has 1s
            for size in range(2, len(indices) + 1):
                for combo in combinations(indices, size):
                    # Create a set representing the subset
                    subset_set = frozenset(combo)
                    
                    # Add the value to this subset key in the dictionary
                    if subset_set in interactions_dict:
                        interactions_dict[subset_set] += value
                    else:
                        interactions_dict[subset_set] = value

        return list(interactions_dict.items())
    


def gemfix_reg(X, y, sample_weight):
    n, d = X.shape
    inner_prod = np.inner(X,X) # X @ X.T
    lam = 0.001 
    Omega = (2 ** inner_prod - 1) + lam * np.diag(sample_weight)

    sample_set_size = np.array(X @ np.ones((d,)), dtype=int)
    size_weight = np.zeros((d,))
    for i in range(1,d+1):
        for j in range(1,i+1):
            size_weight[i-1] += (1/j) * comb(i-1,j-1)
    
    alpha_weight = np.array([size_weight[t-1] if t != 0 else 0 for t in sample_set_size])
    
    L = cholesky(Omega , lower=True)
    alpha = cho_solve((L, True), y)

    shapley_val = np.zeros((d,))
    for i in range(d):
        #shapley_val[i] = (alpha_weight_sv * X_sv[:,i]) @ alpha
        shapley_val[i] = (alpha_weight * X[:,i]) @ alpha

    #print(f"the difference between the two shapley vlaue is {np.linalg.norm(shapley_val - shapley_val2, np.inf)}")    

    ## Compute interactions
    model = lars_path(Omega, y, method='lasso')
    coefs = model[2] ## coefficient of the models for different lambda' values

    # select 6 alpha and add the interacting terms as potential interations of the game
    solution_index = (coefs.shape[1] * np.array([0.05, .1, .2, .3, .4, .5, .7])).astype(int)  # np.array([0.05, .1, .3])).astype(int) #
    unique_interactions = set()

    for ind in solution_index:
        ## add all the possible interactions given the feature size
        interactions = interactions_from_alpha(X, coefs[:,ind])
        intc_topitems = heapq.nlargest(100, interactions, key=lambda x: abs(x[1])) #interactions.sort(key=lambda x: np.abs(x[1]), reverse=True)
        
        for elem in intc_topitems: ## only the first 100 interactions to be added
            unique_interactions.add(elem[0])

    for i in range(d):
        unique_interactions.add(frozenset({i}))
    unique_interactions = list(unique_interactions)
    unique_interactions.sort(key=lambda x: (len(x), (list(x)[0])))

    ## Second, assigning a value for interacting terms: this is done by...
    ## add the interacting terms to the self.mat matrix and run a lasso regression to find their contribution 
    extended_mat = np.zeros( (X.shape[0], len(unique_interactions)) )
    for i, feature_set in enumerate(unique_interactions):
        extended_mat[:,i] = np.prod(X[:,list(feature_set)], axis=1)
    
    model_extended = LassoLarsIC(criterion='bic').fit(extended_mat, y)
    nonzero_index = np.where(abs(model_extended.coef_) > 1e-2)[0] #np.nonzero(model_extended.coef_)[0]
    nonzero_coef = model_extended.coef_[nonzero_index]
    nonzero_interact  = [unique_interactions[i] for i in nonzero_index]

    sparse_coef = list(zip(nonzero_interact, nonzero_coef))

    selected_item = [item for item in nonzero_interact if len(item) > 1] ## only get the interaction effects, not the main ones

    selected_interactions = list(zip(selected_item, model_extended.coef_[nonzero_index]))
    selected_interactions.sort(key=lambda x: abs(x[1]), reverse=True)



    return shapley_val, alpha, selected_interactions

''' 
Estimating Shapley Value with Regression
'''
# Generate the weights for SHAP regression samples 
def generate_weights_by_size(total_players):
    weights_by_size = {}
    for size in range(total_players + 1):  # Include 0 to d
        if size > 0 and total_players - size > 0:  # Valid coalition sizes
            weight = (total_players - 1) / (comb(total_players, size) * size * (total_players - size))
            weights_by_size[size] = weight
        else:  # Handling for empty set and full set
            weights_by_size[size] = 100000  # Assign as needed, e.g., 0
    return weights_by_size

# Building data matrix based on the given subsets
def subset_to_matrix(N, subsets, subset_values):
    # Linear Regression 
    data_binaryfeature = []
    weights = []
    y = []
    weights_by_size = generate_weights_by_size(N)

    # Fill the matrix with binary representations
    for subset in subsets:
        row = [1 if i in subset else 0 for i in range(1,N+1)]
        weights.append(weights_by_size[len(subset)])
        y.append(subset_values[subset])
        data_binaryfeature.append(row)

    data_binaryfeature = np.array(data_binaryfeature)
    y = np.array(y)
    weights = np.array(weights)

    return data_binaryfeature, y, weights 

# Using regression for estimating SHAP values
def Shapley_regression(data_binaryfeature, y, weights, model_type='linear'):

    if model_type == 'linear':
        model = LinearRegression(fit_intercept=False)
        model.fit(data_binaryfeature, y, sample_weight=weights)
    
    elif model_type == 'shap_reg':
        #model = RidgeCV(cv=5, fit_intercept=False)
        model = Ridge(alpha=0.1)
        model.fit(data_binaryfeature, y, sample_weight=weights)

    return model

# Constructing the data matrix for regression based on the mobius transformation 
def subset_to_matrix_mobius(N, subsets, subsets_all, subset_values):
    matrix_size = len(subsets_all) #2 ** N 
    matrix_mobius = np.zeros((len(subsets), matrix_size)) #[[0 for _ in range(matrix_size)] for _ in range(matrix_size-1)]      
    weights = []
    y = []
    # Fill the matrix
    for row_idx, coalition in enumerate(subsets):
        weights.append(10000) if (len(coalition) == N  or len(coalition) == 0) else weights.append(1)
        y.append(subset_values[coalition])
        for col_idx, subset in enumerate(subsets_all):
            # If the subset is a subset of the coalition, mark as 1
            if subset.issubset(coalition):
                matrix_mobius[row_idx][col_idx] = 1

    return matrix_mobius, np.array(y), np.array(weights)

# Estimating the Mobius transformation of game values based on linear regression
def gemfix_regression(matrix_mobius, y, weights, subsets_all, N, model_type='linear', alpha=1.0):

    if model_type == 'linear':
        model_gemfix = LinearRegression(fit_intercept=True)

    elif model_type == 'ridge':
        #model_mobius = Ridge(alpha=alpha, fit_intercept=False)
        model_gemfix = RidgeCV(cv=5, fit_intercept=True)

    elif model_type == 'lasso':
        model_gemfix = Lasso(alpha=alpha, fit_intercept=True)

    elif model_type == 'lassocv':
        model_gemfix = LassoCV(cv=5, random_state=0, fit_intercept=True)

    elif model_type == 'svr':
        model_gemfix = LinearSVR(fit_intercept=True)

    elif model_type == 'gemfix_reg':
        shapley_values, model = gemfix_reg(matrix_mobius, y, weights)
        return shapley_values, model

    model_gemfix.fit(matrix_mobius, y, sample_weight=weights)
    gemfix_shapely = gemfix_shapley_calculation(subsets_all, model_gemfix.coef_, N)

    return np.array(gemfix_shapely), model_gemfix

# Computing Shapley value based on the Mobius transformation values
def gemfix_shapley_calculation(subsets_all, coef, N):
    shapley_mobius = []
    for i in range(1,N+1):
        subsets_indices = [(index) for index, subset in enumerate(subsets_all) if i in subset]
        subsets_weight = [(1 / len(subset)) for _, subset in enumerate(subsets_all) if i in subset]
        shapley_mobius.append(np.sum(coef.squeeze()[subsets_indices] * subsets_weight))

    return shapley_mobius


if __name__ == '__main__':
    '''
    Testing the methods
    '''
    seed(42)  # For reproducibility

    N = 10 # number of players
    players, subsets_all, subset_values, subset_values_mobius = generate_game(N, 3) # generate a game

    exact_shapley_values = exact_Shapley_value(subsets_all, subset_values, N, players) # computing the exact Shapely value of the game
    print(f"Exact shapley value is: {np.round(exact_shapley_values, 3)}")


    # Subset selection for SV estimation
    num_samples = 200
    drawn_samples = np.min((num_samples, 2 ** N))
    #subsets_all.pop(0)
    subsets = random.sample(subsets_all, drawn_samples) # subsets_all #
    # We need to have the game of all players in the regression analysis just to make sure the sum of Shapley values of features is the predicted value
    if subsets_all[-1] not in subsets: subsets.append(subsets_all[-1])
    if subsets_all[0] not in subsets: subsets.append(subsets_all[0])

    # generate data matrix for linear regression to estimate SHAP value
    data_binaryfeature, y, weights = subset_to_matrix(N, subsets, subset_values)
    model = Shapley_regression(data_binaryfeature, y, weights)
    shap_values = model.coef_
    print(f"SHAP regression values: {np.round(shap_values, 3)}")

    #model_rr = Shapley_regression(data_binaryfeature, y, weights, model_type='ridge')
    #shap_values_rr = model_rr.coef_
    #print(f"SHAP regression values: {np.round(shap_values_rr, 3)}")


    # generate data matrix for linear regression estiamting Mobius transformation of game values
    matrix_mobius, y_mobius, weights = subset_to_matrix_mobius(N, subsets, subsets_all, subset_values)

    shapley_mobius_ksvr, ksvr_model = Shapley_mobius_regression(data_binaryfeature, y, weights, subsets_all, N, model_type='ksvr')
    print(f"KSVR   Shapely value is {np.round(shapley_mobius_ksvr, 3)}")

    shapley_mobius_linear, linear_model = Shapley_mobius_regression(matrix_mobius, y_mobius, weights, subsets_all, N)
    print(f"Linear Shapely value is {np.round(shapley_mobius_linear, 3)}")

    shapley_mobius_ridge, ridge_model = Shapley_mobius_regression(matrix_mobius, y_mobius, weights, subsets_all, N, model_type='ridge')
    print(f"Ridge  Shapely value is {np.round(shapley_mobius_ridge, 3)}")

#    shapley_mobius_lassocv, lassocv_model = Shapley_mobius_regression(matrix_mobius, y_mobius, weights, subsets_all, N, model_type='lassocv')
#    print(f"Lasso    Shapely value is {shapley_mobius_lassocv}")

    shapley_mobius_svr, svr_model = Shapley_mobius_regression(matrix_mobius, y_mobius, weights, subsets_all, N, model_type='svr')
    print(f"SVR    Shapely value is {np.round(shapley_mobius_svr, 3)}")
    

    print("done!")









