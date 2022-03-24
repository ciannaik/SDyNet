import numpy as np
import scipy as sp
import math
from scipy.special import gamma


# Sample from GGP

def GGPrnd(alpha, sigma, tau, T):
    a = 5
    sigma = float(sigma)
    alpha = float(alpha)
    tau = float(tau)

    if sigma < -1e-8:
        rate = np.exp(np.log(alpha) - np.log(-sigma) + sigma * np.log(tau))
        K = np.random.poisson(rate)
        N = np.random.gamma(-sigma, 1 / tau, K)
        N = N[N > 0]
        T = 0
        return N
    # Use a truncated Pareto on [T,a] and on (a, infty) use truncated exponential (see later)
    if sigma > 0:
        lograte = np.log(alpha) - np.log(sigma) - tau * T - np.log(sp.special.gamma(1 - sigma)) + np.log(
            T ** (-sigma) - a ** (-sigma))
        Njumps = np.random.poisson(np.exp(lograte))
        log_N1 = - 1 / sigma * np.log(-(np.random.rand(Njumps) * (a ** sigma - T ** sigma) - a ** sigma) / (
                    a * T) ** sigma)  # Sample from truncated Pareto
    else:
        lograte = np.log(alpha) - tau * T - np.log(sp.special.gamma(1 - sigma)) + np.log(np.log(a) - np.log(T))
        Njumps = np.random.poisson(np.exp(lograte))
        log_N1 = np.random.rand(Njumps) * (np.log(a) - np.log(T)) + np.log(T)
    N1 = np.exp(log_N1)
    ind1 = np.log(np.random.rand(Njumps)) < tau * (T - N1)
    N1 = N1[ind1]

    # Use a truncated exponential on (a,+infty) or (T, infty)
    lograte = np.log(alpha) - tau * a - (1 + sigma) * np.log(a) - np.log(tau) - np.log(sp.special.gamma(1 - sigma))
    Njumps = np.random.poisson(np.exp(lograte))
    log_N2 = np.log(a + np.random.exponential(1 / tau, Njumps))  # Sample from truncated exponential
    ind2 = np.log(np.random.rand(Njumps)) < -(1 + sigma) * (log_N2 - np.log(a))
    N2 = np.exp(log_N2[ind2])
    N = np.concatenate((N1, N2))
    return N