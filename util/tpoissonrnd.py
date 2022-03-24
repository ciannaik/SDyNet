import numpy as np
from scipy.stats import poisson

def tpoissrnd(lam):

    x = np.ones(lam.shape)
    ind = (lam > 1e-5) # below this value, x=1 w. very high proba
    lambda_ind = lam[ind]
    x[ind] = poisson.ppf(np.exp(-lambda_ind) +np.multiply(np.random.uniform(size = lambda_ind.shape),(1 - np.exp(-lambda_ind))), lambda_ind)
    return x
