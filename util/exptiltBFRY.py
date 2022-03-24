import numpy as np

def exptiltBFRY(alpha, sigma, tau, L):

    t = np.float((L * sigma / alpha) ** (1 / sigma))

    # simulate sociabilities from exponentially tilted BFRY(sigma, tau, t)
    g = np.random.gamma(1 - sigma, 1, L)
    unif = np.random.rand(L)
    s = np.multiply(g, np.power(((t + tau) ** sigma) * (1 - unif) + (tau ** sigma) * unif, -1 / sigma))
    return s
