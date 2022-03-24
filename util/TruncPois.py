import numpy as np
from scipy.stats import poisson

def tpoissrnd(lam):
    # lam MUST be an array
    if not np.isscalar(lam):
        x = np.ones(lam.shape)
        ind = lam > 1e-5 # below this value x=1 whp
        #ind = ind[:, 0]
        n_ = np.sum(ind)
        lam_ = lam[ind]
        x[ind] = poisson.ppf(np.exp(-lam_) + np.multiply(np.random.rand(n_), 1 - np.exp(-lam_)), lam_)#[:, 0]
    else:
        ind = lam > 1e-5
        if ind:
            n_ = ind
            # lam_ = lam[ind]
            x = poisson.ppf(np.exp(-lam) + np.random.rand() * (1 - np.exp(-lam)), lam)

    return x
