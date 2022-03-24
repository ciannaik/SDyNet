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

plt.rcParams.update({'font.size':16})

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

settings = {}
settings['nchains'] = 3
settings['keep_old']=0
settings['dt']=1
settings['N_samples'] = N_samples
settings['thin'] = thin

# Choose truncation level L
# epsilon**(sigma/(sigma-1))*sigma*(sigma/((2**sigma)*(tau**(sigma*(sigma-1)))))**(1/(1-sigma**2))*(N_alpha)**(1/(1-sigma**2))

settings['L']=20000
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

f = open('store_reuters.pckl', 'rb')

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
    file = 'store_reuters_results_chain_{}.pckl'.format(i+1)
    f = open(file, 'rb')
    settings_chain, weights_st_chain, logpost_st_chain, phi_st_chain, sigma_st_chain, alpha_st_chain, tau_st_chain, rho_st_chain, rate_chain, epsilon_chain, rate_hyper_chain, gvar_samples_chain, w_rate_samples_chain, hyper_rate_samples_chain, epsilon_samples_chain = pickle.load(f)
    f.close()

    weights_st[i] = weights_st_chain


    logpost_st[i] = logpost_st_chain
    phi_st[i] = phi_st_chain
    alpha_st[i] = alpha_st_chain
    sigma_st[i] = sigma_st_chain
    tau_st[i] = tau_st_chain
    rho_st[i] = rho_st_chain

    rate[i] = rate_chain
    epsilon[i] = epsilon_chain

    rate_hyper[i] = rate_hyper_chain

    gvar_samples[i] = gvar_samples_chain
    w_rate_samples[i] = w_rate_samples_chain
    hyper_rate_samples[i] = hyper_rate_samples_chain
    epsilon_samples[i] = epsilon_samples_chain


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
        plt.ylim([2500000,3100000])
        plt.show()
        plt.tight_layout()

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
    f = open('store_reuters_results_all.pckl', 'wb')
    pickle.dump([settings, stats, stats_noburn, phi_st, sigma_st, alpha_st, tau_st], f)
    f.close()
# Sigma


# Posterior Predictive Degree plot

plot_degreepostpred_real_chains(G_sym, objmcmc_noburn, folder_name_noburn, ndraws=20)
