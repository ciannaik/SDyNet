import numpy as np
import scipy
from scipy.stats import poisson

def k_tpoissrnd(lam,k):
    # Generate random draws from a Poisson truncated at k
    x = np.ones(lam.shape)*k
    ind = (lam > 1e-5) # below this value, x=1 w. very high proba
    lambda_ind = lam[ind]
    k_ind = k[ind]
    n = np.sum(ind)
    naccept = 0
    ntrial = 0


    while naccept < n:
        # First check how many indices we have not gotten samples for
        # lambda_ind = lam[ind]
        # Define proposal distribution
        m = np.maximum(np.ceil(k_ind + 1 - lambda_ind), 0).astype(int)
        y = np.random.poisson(lambda_ind)
        x_ac = y + m

        logaccept = (scipy.special.loggamma(x_ac - m+1) + scipy.special.loggamma(k_ind + 2)
                     -scipy.special.loggamma(x_ac + 1) - scipy.special.loggamma(k_ind + 2 - m))

        u = np.random.uniform(size = len(lambda_ind))


        accept = np.squeeze(np.asarray(np.exp(logaccept)))*(x_ac>=k_ind)
        accept[np.isnan(accept)] = 0

        ind_accept = ind * (u < accept)
        x[ind_accept] = x_ac[ind_accept]

        naccept = naccept + np.sum(ind_accept)
        ntrial = ntrial + 1
        ind_needed = ind * (u>accept)
        ind = ind_needed


    return x
