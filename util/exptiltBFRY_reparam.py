import numpy as np

def exptiltBFRY_reparam(t, sigma, tau, L):

    # Reparametrization taking t = (L*sigma/alpha)^(1/sigma), sigma, tau and L as updates
    # alpha is incorporated into the value of t

    # simulate sociabilities from exponentially tilted BFRY(sigma, tau, t)
    g = np.random.gamma(1 - sigma, 1, L)
    unif = np.random.rand(L)

    # Old version - unstable when t is large (and sigma is <0)
    # s = np.multiply(g, np.power(((t + tau) ** sigma) * (1 - unif) + (tau ** sigma) * unif, -1 / sigma))

    # New version
    s = np.multiply(g, (1/tau)*np.power((((t + tau)/tau) ** sigma) * (1 - unif) + unif, -1 / sigma))

    return s
