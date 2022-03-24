import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
import time
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../data'))
sys.path.insert(1, os.path.join(sys.path[0], '../util'))
sys.path.insert(1, os.path.join(sys.path[0], '../plotting'))

from ggp_dyngraphrnd import ggp_dyngraphrnd
from ggp_sample_C import ggp_sample_C
from ggp_sample_hyperparameters import ggp_sample_hyperparameters
from ggp_sample_interaction_counts import ggp_sample_interaction_counts
from bfry_sample_interaction_counts import bfry_sample_interaction_counts
from ggp_sample_rho import *
from bfry_sample_weights import *
from bfry_sample_u import bfry_sample_u
from scipy.sparse import csr_matrix, triu
from plot_degree_post_pred_sparse import *
from bfry_log_posterior_all import *

# np.warnings.filterwarnings('ignore')

folder_name = 'reddit'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Set the seed
random.seed(1001)

N_Gibbs = 400000
N_burn = 0*N_Gibbs
thin = 4000
N_samples = int((N_Gibbs-N_burn)/thin)
save_results = 1

# Settings
sample_rho = 0
rho = 10

settings = {}
settings['nchains'] = 1
settings['keep_old']=0
settings['dt']=1
settings['N_samples'] = N_samples
settings['thin'] = thin

settings['L']=40000
settings['fromggprnd']=1
settings['onlychain']=0
settings['threshold']=1e-5

settings['gcontrol'] =0
settings['g_a'] = .1
settings['g_b'] = .1

settings['leapfrog'] = {}
settings['leapfrog']['epsilon'] = 0.1
settings['leapfrog']['L'] = 10
settings['leapfrog']['nadapt']=10000

settings['augment'] = 'Exponential'

settings['typegraph'] = 'simple'
settings['nlatent'] = 10
settings['nrho'] = 10
settings['estimate_ncounts']= 0
settings['estimate_weights']= 1
settings['estimate_C']= 1
settings['logposterior']= 1
settings['estimate_alpha']= 1
settings['estimate_sigma']= 1
settings['estimate_tau']= 1
settings['estimate_phi']= 1
settings['estimate_rho']= 0
settings['estimate_hypers'] = max(settings['estimate_alpha'],
                                     settings['estimate_sigma'],
                                     settings['estimate_tau'],
                                     settings['estimate_phi'],
                                     settings['estimate_rho'])
settings['rw_std']={}
settings['rw_std'][0] = 0.02 #alpha
settings['rw_std'][1] = 0.02 #sigma
settings['rw_std'][2] = 0.02 #tau
settings['rw_std'][3] = 0.02 #phi
settings['rw_std'][4] = 10000 # nadapt for hyperparameters

settings['hyper_alpha'] = {}
settings['hyper_alpha'][0] = 0.1
settings['hyper_alpha'][1] = 0.1

settings['hyper_tau'] = {}
settings['hyper_tau'][0]=0.1
settings['hyper_tau'][1]=0.1

settings['hyper_phi'] = {}
settings['hyper_phi'][0]=0.1
settings['hyper_phi'][1]=0.1

settings['hyper_sigma'] = {}
settings['hyper_sigma'][0]=0.1
settings['hyper_sigma'][1]=0.1

settings['hyper_rho'] = {}
settings['hyper_rho'][0]=10
settings['hyper_rho'][1]=1

issimple = 1


####################################################
# Load the dataset
####################################################

# For this dataset consider the full model but fix the rho=0 so that no death is taking place.

f = open('store_reddit.pckl', 'rb')

obj = pickle.load(f)
f.close()

tG = obj[0] # tG[t] is the upper tirangular links matrix for t.
G_sym = obj[-3]


T = len(tG)
K_data = tG[0].shape[0]
K = tG[0].shape[0]

L = settings['L']

N_new = obj[-2]
M = np.zeros((K, T), dtype='int')

for t in range(T):
    M[:, t] = np.sum(N_new[t],0)

ind = [None] * T
trZ = [None] * T
linidxs = [None] * T
for t in range(T):
    G = tG[t]

    # Add in extra nodes with no connections
    csr_matrix.resize(G,(L,L))
    linidxs[t] = np.nonzero(G)
    ind[t] = np.nonzero(G)  # the indices refer to the upper triangle

    trZ[t] = csr_matrix(G)


M_new = np.zeros((L,T),dtype=int)
M_new[:K,:] = M

M = M_new


# obj taking up a lot of space - get rid of it
obj = None
K = L

######################################################
# Initialise structures for each chain
######################################################

# Set up separate chains
nchains = settings['nchains']

weights_st = {}
# pi_st = {}

logpost_st = {}
phi_st = {}
alpha_st = {}
sigma_st = {}
tau_st = {}
rho_st = {}


rate = {}
epsilon = {}

rate_hyper = {}

gvar_samples = {}
w_rate_samples = {}
hyper_rate_samples= {}
epsilon_samples = {}

for i in range(nchains):
    weights_st[i] = [None]*T
    # pi_st[i] = [None]*T
    for k in range(T):
        weights_st[i][k]= np.zeros((K, N_samples), dtype='float16')
        # pi_st[i][k]= np.zeros((K, N_samples), dtype='float16')

    logpost_st[i] = np.zeros((7, N_samples))
    phi_st[i] = np.zeros(N_samples)
    alpha_st[i] = np.zeros(N_samples)
    sigma_st[i] = np.zeros(N_samples)
    tau_st[i] = np.zeros(N_samples)
    rho_st[i] = np.zeros(N_samples)


    gvar_samples[i] = np.zeros((T,N_samples))
    w_rate_samples[i] = np.zeros((T,N_samples))
    hyper_rate_samples[i] = np.zeros(N_samples)
    epsilon_samples[i] = np.zeros((T,N_samples))

####################################################
# Run each chain
####################################################

print('***************\n')
print('START dynamic model\n')
print('***************\n')
print('K={} nodes\nT={} time steps\n'.format(K, T))
print('{} MCMC iterations\n'.format(N_Gibbs))
print('***************\n')

t0 = time.time()

for n in range(nchains):





    # Random initialization of the weights
    weights = np.random.uniform(size=(K, T))


    phi = np.random.uniform(10,100)

    phi = 1000

    counts = np.random.poisson(phi * weights)

    alpha = np.random.uniform(100,400)

    alpha = 400

    sigma = np.random.uniform(0.5,0.9)

    sigma = 0.5

    tau = np.random.uniform(0.5,2)

    tau = 2

    if settings['gcontrol']:
        gvar = np.random.gamma(settings['g_a'], 1 / settings['g_b'])
    else:
        gvar = np.ones(T, dtype='int')


    rate_hyper=np.zeros(N_Gibbs)

    rate = np.zeros((T, N_Gibbs))
    epsilon = settings['leapfrog']['epsilon'] / (K - 1) ** (1 / 4) * np.ones(T)  # Leapfrog stepsize

    print('***************\n')
    print('Starting Chain {}\n'.format(n))
    print('***************\n')


    for i in range(N_Gibbs):
        # Sample new interaction counts

        L = K
        u = bfry_sample_u(weights, alpha, sigma, L, settings)

        [weights, rate[:, i]] = bfry_sample_weights(weights, counts, u, M, epsilon, alpha, sigma, tau, L, phi, gvar,
                                                    settings)

        if (i < settings['leapfrog']['nadapt']) & (i > 0):  # Adapt the stepsize
            adapt_lag = 100
            epsilon = np.exp(np.log(epsilon) + .01 * (np.mean(rate[:, max(0,i-adapt_lag):i], 1) - 0.65))

        L = K
        [counts, accept_counts] = ggp_sample_C(counts, weights, phi, alpha, sigma, tau, L)


        [alpha, sigma, tau, phi, rate_hyper[i]] = ggp_sample_hyperparameters(weights, counts, alpha, sigma, tau, phi,
                                                                             settings)

        if (i < settings['rw_std'][4]) & (i > 0):  # Adapt the stepsize
            adapt_lag = 100
            settings['rw_std'][0] = np.exp(np.log(settings['rw_std'][0]) + .01 * (np.mean(rate_hyper[max(0,i-adapt_lag):i]) - 0.23))
            settings['rw_std'][1] = np.exp(np.log(settings['rw_std'][1]) + .01 * (np.mean(rate_hyper[max(0,i-adapt_lag):i]) - 0.23))
            settings['rw_std'][2] = np.exp(np.log(settings['rw_std'][2]) + .01 * (np.mean(rate_hyper[max(0,i-adapt_lag):i]) - 0.23))
            settings['rw_std'][3] = np.exp(np.log(settings['rw_std'][3]) + .01 * (np.mean(rate_hyper[max(0,i-adapt_lag):i]) - 0.23))

        # Plot log-posterior
        logpost = 0
        if settings['logposterior']:
             logpost = bfry_log_posterior(weights, counts, u, M_new, N_new, N_new, alpha, sigma, tau, phi, rho, gvar,
                                             settings)

        if (i >= N_burn and np.remainder((i - N_burn), thin) == 0):
            indd = int(((i - N_burn) / thin))
            for k in range(T):
                weights_st[n][k][:,indd] = weights[:, k]
                indnotNaN = np.isnan(weights[:, k])==0

            for j in range(7):
                logpost_st[n][j,indd] = logpost[j]
            phi_st[n][indd] = phi
            sigma_st[n][indd] = sigma
            alpha_st[n][indd] = alpha
            tau_st[n][indd] = tau
            gvar_samples[n][:,indd] = gvar
            w_rate_samples[n][:, indd] = rate[:, i]
            hyper_rate_samples[n][indd] = rate_hyper[i]
            epsilon_samples[n][:, indd] = epsilon

        if n == 0:
            if i == 99:
                t1 = (time.time() - t0) * N_Gibbs * nchains / (3600 * 100)
                print('Estimated computation time: {} hours\n'.format(t1))
                # print('Estimated end of computation: %s \n', datestr(now + toc * N_Gibbs/3600/24))
                print('***************\n')

t2 = time.time()

print('***************\n')
print('END dynamic model\n')
print('***************\n')
print('Computation time = {} hours\n'.format((t2-t0) / 3600))
print('***************\n')

for i in range(nchains):
    print('Hypers Acceptance Rate = {} \n'.format(np.mean(hyper_rate_samples[i])))
    print('Weights Acceptance Rate = {} \n'.format(np.mean(w_rate_samples[i])))

##################################################
# Save results for combining chains
##################################################
if save_results:
    f = open('store_reddit_results_chain_1.pckl', 'wb')
    pickle.dump([settings, weights_st, logpost_st, phi_st, sigma_st,
                 alpha_st, tau_st, rho_st, rate, epsilon, rate_hyper,
                 gvar_samples, w_rate_samples, hyper_rate_samples, epsilon_samples], f)
    f.close()

####################################################
# Get some summary statistics
####################################################
# Pack results into an objmcmc object

weights_samples =  {}
for i in range(nchains):
    weights_samples[i] =  np.zeros((K, T, N_samples), dtype='float')
    for k in range(T):
        weights_samples[i][:,k,:] =weights_st[i][k]

stats={}
stats['weights_mean'] = np.zeros((K,T))
stats['weights_std'] = np.zeros((K,T))
stats['weights_05'] = np.zeros((K,T))
stats['weights_95'] = np.zeros((K,T))

weights_st_combined = weights_st[0]
for i in range(1,nchains):
    weights_st_combined = np.concatenate((weights_st_combined,weights_st[i]),axis = -1)

for k in range(T):
    stats['weights_mean'][:, k] = np.mean(weights_st_combined[k],1)
    stats['weights_std'][:, k] = np.std(weights_st_combined[k],1)
    stats['weights_05'][:, k] = np.quantile(weights_st_combined[k], .05, 1)
    stats['weights_95'][:, k] = np.quantile(weights_st_combined[k], .95, 1)

if save_results:
    f = open('store_reddit_results.pckl', 'wb')
    pickle.dump([settings, stats, phi_st, sigma_st, alpha_st, tau_st], f)

    f.close()


for i in range(nchains):
    plt.figure()

    plt.plot(np.arange(1, N_samples * thin + 1, thin), w_rate_samples[i][0,:])
    plt.plot(np.arange(1, N_samples * thin + 1, thin),
             np.divide(np.cumsum(w_rate_samples[i][0,:]), np.arange(1, N_samples + 1)), 'g-')

    # plt.legend('95# credible intervals', 'True value')
    plt.xlabel('MCMC iterations')
    plt.ylabel('w_rate')
    plt.savefig('{}/w_rate.png'.format(folder_name))

    plt.figure()

    plt.plot(np.arange(1, N_samples * thin + 1, thin), epsilon_samples[i][0,:])
    plt.plot(np.arange(1, N_samples * thin + 1, thin),
             np.divide(np.cumsum(epsilon_samples[i][0,:]), np.arange(1, N_samples + 1)), 'g-')

    # plt.legend('95# credible intervals', 'True value')
    plt.xlabel('MCMC iterations')
    plt.ylabel('epsilon')
    plt.savefig('{}/epsilon.png'.format(folder_name))


    plt.figure()
    plt.plot(np.arange(1, N_samples * thin + 1, thin), hyper_rate_samples[i])
    plt.plot(np.arange(1, N_samples * thin + 1, thin),
             np.divide(np.cumsum(hyper_rate_samples[i]), np.arange(1, N_samples + 1)), 'g-')

    # plt.legend('95# credible intervals', 'True value')
    plt.xlabel('MCMC iterations')
    plt.ylabel(r'\rho')
    plt.savefig('{}/hyper_rate.png'.format(folder_name))


# Plot logposterior
if settings['logposterior']:
    for j in range(1):
        plt.figure()
        for i in range(nchains):
            plt.plot(np.arange(1,N_samples*thin+1,thin), logpost_st[i][j,:])
        # for i in range(nchains):
        #     plt.plot(np.arange(1,N_samples*thin+1,thin), np.divide(np.cumsum(logpost_samples[i]),np.arange(1,N_samples*thin+1,thin)), 'g-')

        # plt.legend('95# credible intervals', 'True value')
        plt.xlabel('MCMC iterations')
        plt.ylabel('Log Posterior')
        plt.show()

        plt.savefig('{}/logpost_{}.png'.format(folder_name,j))
        #xlim([0, N_Gibbs.*(log(wsnew) -log(Wst))])


objmcmc = {}
objmcmc['settings']=settings
objmcmc['samples'] = {}
objmcmc['samples']['w'] = weights_samples
objmcmc['samples']['alpha'] = alpha_st
objmcmc['samples']['sigma'] = sigma_st
objmcmc['samples']['tau'] = tau_st
objmcmc['samples']['phi'] = phi_st
objmcmc['samples']['rho'] = rho_st
objmcmc['samples']['gvar'] = gvar_samples


# Posterior predictive degree plot with burn-in removed

weights_samples_noburn = {}
alpha_st_noburn = {}
sigma_st_noburn = {}
tau_st_noburn = {}
phi_st_noburn = {}
rho_st_noburn = {}
gvar_samples_noburn = {}


for i in range(nchains):
    weights_samples_noburn[i] = weights_samples[i][:,:,int(N_samples/2):]
    alpha_st_noburn[i] = alpha_st[i][int(N_samples/2):]
    sigma_st_noburn[i] = sigma_st[i][int(N_samples/2):]
    tau_st_noburn[i] = tau_st[i][int(N_samples/2):]
    phi_st_noburn[i] = phi_st[i][int(N_samples/2):]
    rho_st_noburn[i] = rho_st[i][int(N_samples/2):]
    gvar_samples_noburn[i] = gvar_samples[i][:,int(N_samples/2):]
#
objmcmc_noburn = {}
objmcmc_noburn['settings']=settings
objmcmc_noburn['samples'] = {}
objmcmc_noburn['samples']['w'] = weights_samples_noburn
objmcmc_noburn['samples']['alpha'] = alpha_st_noburn
objmcmc_noburn['samples']['sigma'] = sigma_st_noburn
objmcmc_noburn['samples']['tau'] = tau_st_noburn
objmcmc_noburn['samples']['phi'] = phi_st_noburn
objmcmc_noburn['samples']['rho'] = rho_st_noburn
objmcmc_noburn['samples']['gvar'] = gvar_samples_noburn


folder_name_noburn = folder_name+'/noburn'
if not os.path.exists(folder_name_noburn):
    os.makedirs(folder_name_noburn)

stats_noburn={}
stats_noburn['weights_mean'] = np.zeros((K,T))
stats_noburn['weights_std'] = np.zeros((K,T))
stats_noburn['weights_05'] = np.zeros((K,T))
stats_noburn['weights_95'] = np.zeros((K,T))

weights_st_noburn = weights_st
for i in range(nchains):
    for t in range(T):
            weights_st_noburn[i][t] = weights_st[i][t][:,int(N_samples/2):]

weights_st_combined_noburn = weights_st_noburn[0]
for i in range(1,nchains):
    weights_st_combined_noburn = np.concatenate((weights_st_combined_noburn,weights_st_noburn[i]),axis = -1)

for k in range(T):
    stats_noburn['weights_mean'][:, k] = np.mean(weights_st_combined_noburn[k],1)
    stats_noburn['weights_std'][:, k] = np.std(weights_st_combined_noburn[k],1)
    stats_noburn['weights_05'][:, k] = np.quantile(weights_st_combined_noburn[k], .05, 1)
    stats_noburn['weights_95'][:, k] = np.quantile(weights_st_combined_noburn[k], .95, 1)

if save_results:
    f = open('store_reddit_results.pckl', 'wb')
    pickle.dump([settings, stats, stats_noburn, phi_st, sigma_st, alpha_st, tau_st], f)
    f.close()
# Sigma

sigma_all = np.linspace(0.05, 0.95, 100)
logpost_true = np.zeros(len(sigma_all))

for i in range(len(sigma_all)):
    tu = 0
    logpost_all = bfry_log_posterior(weights, counts, u, M, N_new, N_new, alpha, sigma_all[i], tau, phi, rho,
                                     gvar,
                                     settings)

    logpost_true[i] = logpost_all[0]

plt.figure()

plt.plot(sigma_all, logpost_true)

plt.axvline(x=sigma)
plt.xlabel(r'$\sigma$')
plt.ylabel('Log Posterior')
plt.show()

plt.savefig('{}/logpost_sigma.png'.format(folder_name))

# Alpha

alpha_all = np.linspace(10, 1000, 100)
logpost_true = np.zeros(len(alpha_all))

for i in range(len(sigma_all)):
    tu = 0
    logpost_all = bfry_log_posterior(weights, counts, u, M, N_new, N_new, alpha_all[i], sigma, tau, phi, rho,
                                     gvar,
                                     settings)

    logpost_true[i] = logpost_all[0]

plt.figure()

plt.plot(alpha_all, logpost_true)

plt.axvline(x=alpha)
plt.xlabel(r'$\alpha$')
plt.ylabel('Log Posterior')
plt.show()

plt.savefig('{}/logpost_alpha.png'.format(folder_name))

# Sigma

tau_all = np.linspace(0.1, 5, 100)
logpost_true = np.zeros(len(tau_all))

for i in range(len(sigma_all)):
    tu = 0
    logpost_all = bfry_log_posterior(weights, counts, u, M, N_new, N_new, alpha, sigma, tau_all[i], phi, rho,
                                     gvar,
                                     settings)

    logpost_true[i] = logpost_all[0]

plt.figure()

plt.plot(tau_all, logpost_true)

plt.axvline(x=tau)
plt.xlabel(r'$\tau$')
plt.ylabel('Log Posterior')
plt.show()

plt.savefig('{}/logpost_tau.png'.format(folder_name))

# Sigma

phi_all = np.linspace(10, 5000, 100)
logpost_true = np.zeros(len(phi_all))

for i in range(len(sigma_all)):
    tu = 0
    logpost_all = bfry_log_posterior(weights, counts, u, M, N_new, N_new, alpha, sigma, tau, phi_all[i], rho,
                                     gvar,
                                     settings)

    logpost_true[i] = logpost_all[0]

plt.figure()

plt.plot(phi_all, logpost_true)

plt.axvline(x=phi)
plt.xlabel(r'$\phi$')
plt.ylabel('Log Posterior')
plt.show()

plt.savefig('{}/logpost_phi.png'.format(folder_name))


# Posterior Predictive Degree plot

plot_degreepostpred_real_chains(G_sym, objmcmc_noburn, folder_name_noburn, ndraws=10)
