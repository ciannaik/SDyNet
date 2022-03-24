import numpy as np
from scipy.stats import binom

def tbinomrnd(n,p,size=None):
    if size is None:
        if np.isscalar(n):
            if not np.isscalar(p):
                x_size = p.shape
            else: x_size = 1
        else:
            if np.isscalar(p):
                x_size = n.shape
            else:
                if n.size == p.size:
                    x_size = n.shape
                else:
                    return ValueError('shape mismatch: objects cannot be broadcast to a single shape')
    else:
        x_size = size

    u = np.random.uniform(size=x_size)
    x = binom.ppf((1-binom.cdf(0,n,p))*u + binom.cdf(0,n,p),n,p)
    return x
