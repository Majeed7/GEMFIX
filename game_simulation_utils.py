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

# Generate weights used for computing Shapley value
# def generate_Shapley_weights_by_size(N):
#     N_fact = factorial(N)

#     # Array to store weights for each subset size, from 0 to N
#     weights = [0] * (N + 1)
    
#     for size in range(N):
#         # Compute weight based on the size of the subset
#         if size == 0 or size == N:
#             # Handle the empty set and the full set cases if necessary
#             # For Shapley value calculation, these cases are usually not directly used
#             # but they are included here for completeness
#             weights[size] = 1 / N
#         else:
#             weights[size] = (factorial(size) * factorial(N - size - 1)) / N_fact
    
#     return weights


# def exact_Shapley_value(subsets_all, subset_values, N, players):
#     # Initialize Shapley value estimation
#     shapley_values = {player: 0 for player in players}
#     shapley_weights = generate_Shapley_weights_by_size(N)

#     # Estimate Shapley values
#     for player in players:
#         for subset in subsets_all:
#             if player not in subset:
#                 subset_with_player = subset.union([player])
#                 if subset_with_player in subset_values:
#                     marginal_contribution = subset_values[subset_with_player] - subset_values.get(subset, 0)
#                     shapley_values[player] += shapley_weights[len(subset)] * marginal_contribution

#     return list(shapley_values.values())


# def Omega(X,i):
#     n, d = X.shape
    
#     idx = np.arange(d)
#     idx[i] = 0
#     idx[0] = i
#     X = X[:,idx]
    
#     omega = np.zeros((n,))
#     ind_nonzeros = np.where(X[:,0] > 0)[0].tolist()
#     for i in ind_nonzeros:
#         xi_ones = np.where(X[i,1:] > 0)[0].tolist()
#         xi_ones_count = len(xi_ones)
#         temp = 0
#         for j in range(1,d):
#             temp += (1 / (j+1)) * (comb(xi_ones_count,j)) 
        
#         omega[i] = temp 
#     omega[ind_nonzeros] = (1 + omega[ind_nonzeros])
#     return omega
    
def gemfix_reg(X, y, sample_weight):
    n, d = X.shape
    inner_prod = np.inner(X,X) # X @ X.T
    kernel_mat = 2 ** inner_prod - 1

    sample_set_size = np.array(X @ np.ones((d,)), dtype=int)
    size_weight = np.zeros((d,))
    for i in range(1,d+1):
        for j in range(1,i+1):
            size_weight[i-1] += (1/j) * comb(i-1,j-1)
    
    alpha_weight = np.array([size_weight[t-1] if t != 0 else 0 for t in sample_set_size])
    
    
    lam = 0.001 
    L = cholesky(kernel_mat + lam * np.diag(sample_weight) , lower=True)
    alpha = cho_solve((L, True), y)

    shapley_val = np.zeros((d,))
    for i in range(d):
        #shapley_val[i] = (alpha_weight_sv * X_sv[:,i]) @ alpha
        shapley_val[i] = (alpha_weight * X[:,i]) @ alpha

    #print(f"the difference between the two shapley vlaue is {np.linalg.norm(shapley_val - shapley_val2, np.inf)}")    

    return shapley_val, alpha

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

class GEMFIX(BaseEstimator, RegressorMixin):
    """Game Estiamtion of Mobius representation for feature interaction detection and explanation

    Parameters
    ----------
    lam : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to lam. Must be strictly positive.

    kernel : {'linear', 'rbf'}, default='linear'
        Specifies the kernel type to be used in the algorithm.
        It must be 'linear', 'rbf' or a callable.

    gamma : float, default = None
        Kernel coefficient for 'rbf'


    Attributes
    ----------
    support_: boolean np.array of shape (n_samples,), default = None
        Array for support vector selection.

    alpha_ : array-like
        Weight matrix

    bias_ : array-like
        Bias vector


    """

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, v, sample_weight):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        support : boolean np.array of shape (n_samples,), default = None
            Array for support vector selection.

        Returns
        -------
        self : object
            An instance of the estimator.
        """
        self.X_train = X 

        X, v = check_X_y(X, v, multi_output=True, dtype='float')

        n, d = X.shape
        inner_prod = np.inner(X,X) # X @ X.T
        self.omega = 2 ** inner_prod - 1

        lam = 1 
        self.L = cholesky(self.omega + lam * np.diag(sample_weight) , lower=True)
        self.alpha = cho_solve((self.L, True), v)

        return self

    def predict(self, X):
        """
        Predict using the estimator.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """

        X = check_array(X, ensure_2d=False)
        inner_prod = np.inner(self.X_train,X) # X @ X.T
        omega = 2 ** inner_prod - 1
 
        return (K @ self.alpha_) + self.bias_

    def score(self, X, y):
        from scipy.stats import pearsonr
        p, _ = pearsonr(y, self.predict(X))
        return p ** 2

    def norm_weights(self):
        A = self.alpha_.reshape(-1, 1) @ self.alpha_.reshape(-1, 1).T

        W = A @ self.K_[self.support_, :]
        return np.sqrt(np.sum(np.diag(W)))


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









