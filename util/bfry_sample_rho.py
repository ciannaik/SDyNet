import numpy as np
from linear_slice_step import linear_slice_step


def compute_lpdf(r, nmt, no, dt):
    w = len(nmt)
    out = 0
    for t in range(1, w):
        out = out + np.sum(np.multiply(-r * dt, no[t]) + np.multiply((nmt[t-1] - no[t]), np.log(1 - np.exp(-r * dt))))
    return out


def log_f_func(y, nmt, no, rho_a, rho_b, dt):
    out = compute_lpdf(np.exp(y), nmt, no, dt)
    out = out + rho_a * np.log(rho_b) - rho_b * np.exp(y) + rho_a * y
    return out


def bfry_sample_rho(rho, nold, nnew, settings):
    # Function that implements the slice sampler for parameter rho
    dt = settings['dt']
    rho_a = settings['hyper_rho'][0]
    rho_b = settings['hyper_rho'][1]

    T = len(nnew)
    n = [None]*T
    for t in range(T):
        n[t] = nnew[t] + nold[t]

    log_f_curr = log_f_func(np.log(rho), n, nold, rho_a, rho_b, dt)
    res, _ = linear_slice_step(np.log(rho), log_f_curr, log_f_func, n, nold, rho_a, rho_b, dt, slice_width=10,
                              prng=np.random, max_steps_out=0, max_slice_iters=200)
    rho_out = np.exp(res)

    return rho_out
