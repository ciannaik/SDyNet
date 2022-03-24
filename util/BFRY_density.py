import numpy as np
import scipy

def dBFRY(w, alpha, sigma, tau, L):
    t_BFRY = np.float((L * sigma / alpha) ** (1 / sigma))

    # simulate sociabilities from exponentially tilted BFRY(sigma, tau, t)
    d = sigma*w**(-1-sigma)*np.exp(-tau*w)*(1-np.exp(-t_BFRY*w))/(scipy.special.gamma(1-sigma)*((tau+t_BFRY)**sigma-tau**sigma))
    return d
