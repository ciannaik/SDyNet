import numpy as np
from texprnd import texprnd
import TruncPois


# Update of the weights
def bfry_sample_u(weights, alpha, sigma, L_BFRY, settings):

    method = settings['augment']
    # L_BFRY = settings['L']

    # Update u directly from true distribution
    t_BFRY = np.float((L_BFRY * sigma / alpha) ** (1 / sigma))

    if method == 'Poisson':
        rate = t_BFRY*weights
        u_out = TruncPois.tpoissrnd(rate)
    elif method == 'Exponential':
        u_out = texprnd(weights, t_BFRY)
    else:
        raise ValueError('Method: "{}" Not Implmented'.format(method))


    return u_out
