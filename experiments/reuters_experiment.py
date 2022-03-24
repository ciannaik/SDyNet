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
folder_name = 'reuters'
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
tau = 1

settings = {}
settings['nchains'] = 1

settings['keep_old']=0
settings['dt']=1
settings['N_samples'] = N_samples
settings['thin'] = thin

# Choose truncation level L

settings['L']=20000
settings['fromggprnd']=1
settings['onlychain']=0
settings['threshold']=1e-4

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
settings['estimate_ncounts']= 1
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

f = open('store_reuters.pckl', 'rb')
obj = pickle.load(f)
f.close()

tG = obj[0] # tG[t] is the upper tirangular links matrix for t.
G_sym = obj[-3]


T = len(tG)
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

####################################################
# Initialise parameters and structures
####################################################
weights = np.random.uniform(size=(K,T))  # random initialization of the weights

phi = np.random.gamma(settings['hyper_phi'][0], 1 / settings['hyper_phi'][1])
phi = np.random.uniform([20,50])

counts = np.random.poisson(phi * weights)

alpha = np.random.gamma(settings['hyper_alpha'][0], 1 / settings['hyper_alpha'][1])

alpha = np.random.uniform([50,200])

sigma = np.random.gamma(settings['hyper_sigma'][0], 1 / settings['hyper_sigma'][1])

sigma = np.random.uniform([0.1,0.5])

if settings['gcontrol']:
    gvar = np.random.gamma(settings['g_a'], 1 / settings['g_b'])
else:
    gvar = np.ones(T,dtype='int')



weights_st = [None]*T
pi_st = [None]*T
for k in range(T):
    weights_st[k]= np.zeros((K, N_samples), dtype='float16')
    pi_st[k]= np.zeros((K, N_samples), dtype='float16')

logpost_st = np.zeros((7, N_samples))
phi_st = np.zeros(N_samples)
alpha_st = np.zeros(N_samples)
sigma_st = np.zeros(N_samples)
tau_st = np.zeros(N_samples)


rate=np.zeros((T,N_Gibbs))
epsilon = settings['leapfrog']['epsilon']/(K-1)**(1/4)*np.ones(T) # Leapfrog stepsize

rate_hyper=np.zeros(N_Gibbs)

gvar_samples = np.zeros((T,N_samples))
w_rate_samples = np.zeros((T,N_samples))
hyper_rate_samples = np.zeros(N_samples)
epsilon_samples = np.zeros((T,N_samples))
####################################################
# Initialize interaction counts
####################################################

dt = settings['dt']


print('***************\n')
print('START dynamic model\n')
print('***************\n')
print('K={} nodes\nT={} time steps\n'.format(K, T))
print('{} MCMC iterations\n'.format(N_Gibbs))
print('***************\n')

rate_hyper=np.zeros(N_Gibbs)

t0 = time.time()



for i in range(N_Gibbs):

    # Sample new interaction counts


    L = K
    u = bfry_sample_u(weights, alpha, sigma, L, settings)

    [weights, rate[:, i]] = bfry_sample_weights(weights, counts, u, M, epsilon, alpha, sigma, tau, L, phi, gvar,
                                                settings)

    if (i < settings['leapfrog']['nadapt']) & (i > 0):  # Adapt the stepsize
        epsilon = np.exp(np.log(epsilon) + .01 * (np.mean(rate[:, :i], 1) - 0.65))

    L = K
    [counts, accept_counts] = ggp_sample_C(counts, weights, phi, alpha, sigma, tau, L)


    [alpha, sigma, tau, phi, rate_hyper[i]] = ggp_sample_hyperparameters(weights, counts, alpha, sigma, tau, phi,
                                                                         settings)

    if (i < settings['rw_std'][4]) & (i > 0):  # Adapt the stepsize
        settings['rw_std'][0] = np.exp(np.log(settings['rw_std'][0]) + .01 * (np.mean(rate_hyper[:i]) - 0.23))
        settings['rw_std'][1] = np.exp(np.log(settings['rw_std'][1]) + .01 * (np.mean(rate_hyper[:i]) - 0.23))
        settings['rw_std'][2] = np.exp(np.log(settings['rw_std'][2]) + .01 * (np.mean(rate_hyper[:i]) - 0.23))
        settings['rw_std'][3] = np.exp(np.log(settings['rw_std'][3]) + .01 * (np.mean(rate_hyper[:i]) - 0.23))


    # Plot log-posterior
    logpost = np.zeros(7)
    if settings['logposterior']:
        logpost = bfry_log_posterior(weights, counts, u, M_new, N_new, N_new, alpha, sigma, tau, phi, rho, gvar,
                                     settings)

    if (i >= N_burn and np.remainder((i - N_burn), thin) == 0):
        indd = int(((i - N_burn) / thin))
        for k in range(T):
            weights_st[k][:,indd] = weights[:, k]
            indnotNaN = np.isnan(weights[:, k])==0
            pi_st[k][:,indd] = weights[:, k]/np.sum(weights[indnotNaN, k])

        for j in range(7):
            logpost_st[j, indd] = logpost[j]
        phi_st[indd] = phi
        sigma_st[indd] = sigma
        alpha_st[indd] = alpha
        tau_st[indd] = tau
        gvar_samples[:,indd] = gvar
        w_rate_samples[:, indd] = rate[:, i]
        hyper_rate_samples[indd] = rate_hyper[i]
        epsilon_samples[:, indd] = epsilon

    if i==99:
        t1 = (time.time()-t0) * N_Gibbs / (3600*100)
        print('Estimated computation time: {} hours\n'.format(t1))
        # print('Estimated end of computation: %s \n', datestr(now + toc * N_Gibbs/3600/24))
        print('***************\n')




t2 = time.time()

print(t2-t0)
print('***************\n')
print('END dynamic model\n')
print('***************\n')
print('Computation time = {} hours\n'.format((t2-t0) / 3600))
print('***************\n')

##################################################
# Save results for combining chains
##################################################
if save_results:
    rho_st = np.zeros(N_samples)
    f = open('store_reuters_results_chain_1.pckl', 'wb')
    pickle.dump([settings, weights_st, logpost_st, phi_st, sigma_st,
                 alpha_st, tau_st, rho_st, rate, epsilon, rate_hyper,
                 gvar_samples, w_rate_samples, hyper_rate_samples, epsilon_samples], f)
    f.close()

####################################################
# Get some summary statistics
####################################################
# Pack results into an objmcmc object

weights_samples =  np.zeros((K, T, N_samples), dtype='float')
for k in range(T):
    weights_samples[:,k,:] =weights_st[k]

objmcmc = {}
objmcmc['settings']=settings
objmcmc['samples'] = {}
objmcmc['samples']['w'] = weights_samples
objmcmc['samples']['alpha'] = alpha_st
objmcmc['samples']['sigma'] = sigma_st
objmcmc['samples']['tau'] = tau_st
objmcmc['samples']['phi'] = phi_st
objmcmc['samples']['rho'] = np.zeros(N_samples)
objmcmc['samples']['gvar'] = gvar_samples




# Posterior Predictive Degree plot
plot_degreepostpred_real(G_sym, objmcmc, folder_name, ndraws=10)



stats={}
stats['weights_mean'] = np.zeros((K,T))
stats['weights_std'] = np.zeros((K,T))
stats['weights_05'] = np.zeros((K,T))
stats['weights_95'] = np.zeros((K,T))
for k in range(T):
    stats['weights_mean'][:, k] = np.mean(weights_st[k],1)
    stats['weights_std'][:, k] = np.std(weights_st[k],1)
    stats['weights_05'][:, k] = np.quantile(weights_st[k], .05, 1)
    stats['weights_95'][:, k] = np.quantile(weights_st[k], .95, 1)

if save_results:
    f = open('store_reuters_results.pckl', 'wb')
    pickle.dump([settings, stats, phi_st, sigma_st, alpha_st, tau_st], f)
    f.close()

plt.figure()

plt.plot(np.arange(1, N_samples * thin + 1, thin), w_rate_samples[0,:])
plt.plot(np.arange(1, N_samples * thin + 1, thin),
         np.divide(np.cumsum(w_rate_samples[0,:]), np.arange(1, N_samples * thin + 1, thin)), 'g-')

# plt.legend('95# credible intervals', 'True value')
plt.xlabel('MCMC iterations')
plt.ylabel('w_rate')
plt.show()

plt.figure()

plt.plot(np.arange(1, N_samples * thin + 1, thin), epsilon_samples[0,:])
plt.plot(np.arange(1, N_samples * thin + 1, thin),
         np.divide(np.cumsum(epsilon_samples[0,:]), np.arange(1, N_samples * thin + 1, thin)), 'g-')

# plt.legend('95# credible intervals', 'True value')
plt.xlabel('MCMC iterations')
plt.ylabel('epsilon')
plt.show()

plt.figure()
plt.plot(np.arange(1, N_samples * thin + 1, thin), hyper_rate_samples)
plt.plot(np.arange(1, N_samples * thin + 1, thin),
         np.divide(np.cumsum(hyper_rate_samples), np.arange(1, N_samples * thin + 1, thin)), 'g-')

# plt.legend('95# credible intervals', 'True value')
plt.xlabel('MCMC iterations')
plt.ylabel(r'\rho')
plt.show()