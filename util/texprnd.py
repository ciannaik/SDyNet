import numpy as np

def texprnd(w,t):

    # Rate parameter w
    # Truncated on [0,t]

    u = np.random.uniform(size=w.shape)

    # x = -(1/w)*np.log(1-(1-np.exp(-t*w))*u)
    # np.exp(-t*w) overflows to 1 for very small values of t*w
    # use a Taylor approximation in this case

    w_large = w[w >= 1e-8 / t]
    w_small = w[w < 1e-8 / t]

    u_large = u[w >= 1e-8 / t]
    u_small = u[w < 1e-8 / t]

    x_large = -(1/w_large)*np.log(1-(1-np.exp(-t*w_large))*u_large)
    x_small = -(1/w_small)*np.log(1-(t * w_small - 0.5 * (t * w_small) ** 2)*u_small)

    x = np.zeros(w.shape)
    x[w >= 1e-8 / t] = x_large
    x[w < 1e-8 / t] = x_small

    return x
