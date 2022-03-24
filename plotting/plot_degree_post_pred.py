import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../experiments'))
sys.path.insert(1, os.path.join(sys.path[0], '../data'))
sys.path.insert(1, os.path.join(sys.path[0], '../util'))
from ggp_dyngraphrnd import ggp_dyngraphrnd
from ggp_sample_C import ggp_sample_C
from ggp_sample_hyperparameters import ggp_sample_hyperparameters
from ggp_sample_interaction_counts import ggp_sample_interaction_counts
from bfry_sample_interaction_counts import bfry_sample_interaction_counts
from ggp_sample_rho import *
from bfry_sample_weights import *
from bfry_sample_u import bfry_sample_u
from scipy.sparse import csr_matrix, triu
from plot_degree import plot_degree

def plot_degreepostpred(G, objmcmc, folder_name, ndraws=5000):

    # plot_degreepostpred
    # 1) plots the degree distribution from the posterior predictive distribution
    #    either conditioning on the hyperparameters only (cond = false)or
    #    using the estimated parameter values (cond = true)
    # -------------------------------------------------------------
    # INPUTS
    #   - G: undirected binary adjacency matrix
    #   - samples: MCMC output
    #
    # OPTIONAL INPUT
    #   - ndraws: number of draws from the posterior predictive
    #   - cond: indicates whether the graphs will be sampled from the posterior
    #   of hyperparameters (cond = false) or parameters (cond = true)
    #   - rep:  output directory
    #   - prefix: output filename prefix
    #   - suffix: output filename suffix
    #   - verbose: logical to indicate whether to print progress info

    # OUTPUT
    #   - quantile_freq:  quantiles of the of the values in data G for the cumulative probability distribution (for matrix G)
    #   - quantile_freq2: quantiles of the of the values in data G for the cumulative probability distribution (for matrix G')

    #--------------------------------------------------------------------------


    samples = objmcmc['samples']
    nsamples = samples['w'].shape[2]
    T = samples['w'].shape[1]
    ind = [math.floor(n) for n in np.linspace(0, nsamples-1, num=ndraws)]
    settings = objmcmc['settings']

    freq_samp = np.zeros((T,ndraws, 17))
    centerbins1 = np.zeros((T,17))
    freq_true = np.zeros((T,17))

    quantile_freq = np.zeros((T,2,17))
    htemp = plt.figure()

    tstart = time.time()


    for i in range(ndraws):

        if np.mod(i, 10) == 0:
            print(i)

        alpha = samples['alpha'][ind[i]]
        sigma = samples['sigma'][ind[i]]
        tau = samples['tau'][ind[i]]
        phi = samples['phi'][ind[i]]
        gvar = samples['gvar'][:,ind[i]]
        rho = samples['rho'][ind[i]]

        [Gsamp, _, _, _, _, _, _, _, _, _] = ggp_dyngraphrnd('BFRY', alpha, sigma, tau, T, phi,
                                                                                 rho, gvar, settings)


        for t in range(T):
            [_, _, freq_samp[t, i, :]] = plot_degree(Gsamp[t,:,:])


    for t in range(T):
        [_, centerbins1[t,:], freq_true[t,:]] = plot_degree(G[t,:,:])

    plt.close()

    for t in range(T):
        quantile_freq[t,:,:] = plot_figure(freq_samp[t,:,:], centerbins1[t,:], freq_true[t,:], t)
        plt.savefig('{}/posterior_predictive_T_{}.png'.format(folder_name, t))


    return quantile_freq

def plot_degreepostpred_real(G, objmcmc, folder_name, ndraws=5000):

    # plot_degreepostpred
    # 1) plots the degree distribution from the posterior predictive distribution
    #    either conditioning on the hyperparameters only (cond = false)or
    #    using the estimated parameter values (cond = true)
    # -------------------------------------------------------------
    # INPUTS
    #   - G: undirected binary adjacency matrix
    #   - samples: MCMC output
    #
    # OPTIONAL INPUT
    #   - ndraws: number of draws from the posterior predictive
    #   - cond: indicates whether the graphs will be sampled from the posterior
    #   of hyperparameters (cond = false) or parameters (cond = true)
    #   - rep:  output directory
    #   - prefix: output filename prefix
    #   - suffix: output filename suffix
    #   - verbose: logical to indicate whether to print progress info

    # OUTPUT
    #   - quantile_freq:  quantiles of the of the values in data G for the cumulative probability distribution (for matrix G)
    #   - quantile_freq2: quantiles of the of the values in data G for the cumulative probability distribution (for matrix G')

    #--------------------------------------------------------------------------


    samples = objmcmc['samples']
    nsamples = samples['w'].shape[2]
    T = samples['w'].shape[1]
    ind = [math.floor(n) for n in np.linspace(0, nsamples-1, num=ndraws)]
    settings = objmcmc['settings']

    freq_samp = np.zeros((T,ndraws, 17))
    centerbins1 = np.zeros((T,17))
    freq_true = np.zeros((T,17))

    quantile_freq = np.zeros((T,2,17))
    htemp = plt.figure()

    tstart = time.time()


    for i in range(ndraws):

        if np.mod(i, 1) == 0:
            print(i)

        alpha = samples['alpha'][ind[i]]
        sigma = samples['sigma'][ind[i]]
        tau = samples['tau'][ind[i]]
        phi = samples['phi'][ind[i]]
        gvar = samples['gvar'][:,ind[i]]
        rho = samples['rho'][ind[i]]

        [Gsamp, _, _, _, _, _, _, _, _, _] = ggp_dyngraphrnd('BFRY', alpha, sigma, tau, T, phi,
                                                                                 rho, gvar, settings)


        for t in range(T):
            [_, _, freq_samp[t, i, :]] = plot_degree(Gsamp[t,:,:])


    for t in range(T):
        [_, centerbins1[t,:], freq_true[t,:]] = plot_degree(G[t].todense())

    plt.close()

    for t in range(T):
        quantile_freq[t,:,:] = plot_figure(freq_samp[t,:,:], centerbins1[t,:], freq_true[t,:], t)
        plt.savefig('{}/posterior_predictive_T_{}.png'.format(folder_name,t))



    return quantile_freq



def plot_figure(freq, centerbins, freq_true, t):

    quantile_freq = np.quantile(freq, [.025, .975],0)
    ind1 = quantile_freq[0,:]==0
    quantile_freq[0, ind1] = quantile_freq[1, ind1] / 100000

    plt.figure()

    plt.plot(centerbins, np.transpose(quantile_freq), color='b', alpha=0.2, label='_nolegend_')
    ind = quantile_freq[0,:]>0

    plt.fill_between(centerbins[ind], np.transpose(quantile_freq[0,ind]),np.transpose(quantile_freq[1,ind]),alpha=0.2)
    plt.xscale('log')
    plt.yscale('log')

    ind = freq_true>0

    plt.loglog(centerbins[ind], freq_true[ind], 'o', color='b')


    plt.legend(labels=('Data', '95% posterior predictive'),frameon=False)
    plt.title('T={}'.format(t))

    plt.xlim([.8, 1e4])


    return quantile_freq