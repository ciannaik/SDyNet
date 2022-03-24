import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import time
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../data'))
sys.path.insert(1, os.path.join(sys.path[0], '../util'))
sys.path.insert(1, os.path.join(sys.path[0], '../plotting'))

from ggp_dyngraphrnd_sparse import ggp_dyngraphrnd_sparse
from ggp_sample_C import ggp_sample_C
from ggp_sample_hyperparameters import ggp_sample_hyperparameters
from ggp_sample_interaction_counts import ggp_sample_interaction_counts
from bfry_sample_interaction_counts import bfry_sample_interaction_counts
from ggp_sample_rho import *
from bfry_sample_weights import *
from bfry_sample_u import bfry_sample_u
from ggp_sample_gvar_mh import ggp_sample_gvar_mh
from scipy.sparse import csr_matrix, triu, diags, lil_matrix, coo_matrix
from plot_degree import plot_degree
from plot_degree_post_pred_sparse import *
from bfry_log_posterior_all import *


plt.rcParams.update({'font.size':16})

# np.warnings.filterwarnings('ignore')

# Folder to save results to

folder_name = 'simulated'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Set the seed
random.seed(1001)

talpha = 100
tsigma = 0.2
ttau = 1 # Parameters gamma process
tphi = 10           # tunes dependence in dependent gamma process
T = 4                        # Number of time steps
trho = 0                     # death rate for latent interactions

settings = {}

settings['dt']=1

settings['L']=15000
settings['fromggprnd']=1
settings['nchains'] = 3
settings['threshold']=1e-5
settings['keep_old']=0


settings['gcontrol'] = 0
tgvar = np.ones(T,dtype='int')
# tgvar = np.array([0.5,0.8,1.2,1.5])
settings['g_a'] = .1
settings['g_b'] = .1

settings['leapfrog'] = {}
settings['leapfrog']['epsilon'] = 0.1
settings['leapfrog']['L'] = 10
settings['leapfrog']['nadapt']=20000

settings['augment'] = 'Exponential'


N_Gibbs = 600000
N_burn = 0*N_Gibbs
thin = 6000
N_samples = int((N_Gibbs-N_burn)/thin)

settings['typegraph'] = 'simple'
settings['nlatent'] = 10
settings['ngvar'] = 10
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
settings['rw_std'][4] = 20000 # nadapt for hyperparameters

settings['hyper_alpha'] = {}
settings['hyper_alpha'][0] = 0
settings['hyper_alpha'][1] = 0

settings['hyper_tau'] = {}
settings['hyper_tau'][0]=0
settings['hyper_tau'][1]=0

settings['hyper_phi'] = {}
settings['hyper_phi'][0]=0
settings['hyper_phi'][1]=0

settings['hyper_sigma'] = {}
settings['hyper_sigma'][0]=0
settings['hyper_sigma'][1]=0

settings['hyper_rho'] = {}
settings['hyper_rho'][0]=10
settings['hyper_rho'][1]=1


if settings['fromggprnd']:
    method = 'GGP'
else:
    method = 'BFRY'

f = open('store_simulated.pckl', 'rb')
Z, w, c, KT,  N_new, N_old, Nall, M, indchain, indcnz = pickle.load(f)
f.close()

print(Z[0].shape)
print(sum(M))
print(sum(indchain))
print([np.sum(np.sum(z,1)>0) for z in Z])


[T, K] = w.shape

L = K

tindchain = indchain.copy()
# indchain = np.full(len(indchain), True)

indlog = np.full(len(indchain), False)
indlog[indchain] = True


#####################################################
# Set some random indices of indlog to be true
indlog_extra = np.full(len(indlog[~indchain]), False)
indchain_extra = random.sample(population=range(len(indlog_extra)),k=min(settings['L']-sum(indchain),sum(~indchain)))
indlog_extra[indchain_extra] = True

indlog[~indchain] = indlog_extra
indchain = indlog
#####################################################


tw= np.transpose(w[:,indlog])
tindchain_trunc = tindchain[indlog]
indchain_trunc = indchain[indlog]
[K ,T] =tw.shape

tc= np.transpose(c[:,indlog])

tm = M[indchain, :]


tNnew = [None]*T
tNold = [None]*T
tN = [None]*T

tmnnew = np.zeros((K, T), dtype='int')
tmnold = np.zeros((K, T), dtype='int')
tmnall = np.zeros((K, T), dtype='int')

tn_new = [None] * T
tn_old = [None] * T

for t in range(T):
    Z[t] = Z[t][indchain, :][:, indchain]

    # make the N matrices symmetric
    N_new_t = N_new[t][indchain, :][:, indchain]
    tNnew[t] = N_new_t + np.transpose(N_new_t) - diags(N_new_t.diagonal())
    tmnnew[:, t] = tm[:, t]
    tn_new[t] = triu(tNnew[t], 1)

    N_old_t = N_old[t][indchain, :][:, indchain]
    tNold[t] = N_old_t + np.transpose(N_old_t) - diags(N_old_t.diagonal())
    tmnold[:, t] = np.ndarray.flatten(np.array(tNold[t].sum(axis=0)))

    N_t = Nall[t][indchain, :][:, indchain]
    tN[t] = N_t + np.transpose(N_t) - diags(N_t.diagonal())
    tmnall[:, t] = np.ndarray.flatten(np.array(tN[t].sum(axis=0)))

######################################################
# Initialise structures for each chain
######################################################

# Set up separate chains
nchains = settings['nchains']

weights_samples = {}
# pi_st = {}
c_samples = {}


logpost_samples = {}
phi_samples = {}
alpha_samples = {}
sigma_samples = {}
tau_samples = {}
rho_samples = {}


rate = {}
epsilon = {}

rate_hyper = {}

gvar_samples = {}
w_rate_samples = {}
hyper_rate_samples= {}
epsilon_samples = {}

for i in range(nchains):
    file = 'store_simulated_results_chain_{}.pckl'.format(i+1)
    f = open(file, 'rb')
    settings_chain, weights_st_chain, c_st_chain, logpost_st_chain, phi_st_chain, sigma_st_chain, alpha_st_chain, tau_st_chain, rho_st_chain, rate_chain, epsilon_chain, rate_hyper_chain, gvar_samples_chain, w_rate_samples_chain, hyper_rate_samples_chain, epsilon_samples_chain = pickle.load(f)
    f.close()

    weights_samples[i] = weights_st_chain[0]
    c_samples[i] = c_st_chain[0]


    logpost_samples[i] = logpost_st_chain[0]
    phi_samples[i] = phi_st_chain[0]
    alpha_samples[i] = alpha_st_chain[0]
    sigma_samples[i] = sigma_st_chain[0]
    tau_samples[i] = tau_st_chain[0]
    rho_samples[i] = rho_st_chain[0]

    rate[i] = rate_chain[0]
    epsilon[i] = epsilon_chain[0]

    rate_hyper[i] = rate_hyper_chain[0]

    gvar_samples[i] = gvar_samples_chain[0]
    w_rate_samples[i] = w_rate_samples_chain[0]
    hyper_rate_samples[i] = hyper_rate_samples_chain[0]
    epsilon_samples[i] = epsilon_samples_chain[0]


####################################################
# Get some summary statistics
####################################################
# Pack results into an objmcmc object

objmcmc = {}
objmcmc['settings']=settings
objmcmc['samples'] = {}
objmcmc['samples']['w'] = weights_samples
objmcmc['samples']['c'] = c_samples
objmcmc['samples']['logpost'] = logpost_samples
objmcmc['samples']['alpha'] = alpha_samples
objmcmc['samples']['sigma'] = sigma_samples
objmcmc['samples']['tau'] = tau_samples
objmcmc['samples']['phi'] = phi_samples
objmcmc['samples']['gvar'] = gvar_samples
objmcmc['samples']['rho'] = rho_samples


indt=[None]*T
for t in range(T):
    indt[t] = np.argsort(tm[:,t])[::-1] # Sort in decreasing order


Na = tm.shape[0]


weights_samples_noburn = {}
c_samples_noburn = {}

alpha_samples_noburn = {}
sigma_samples_noburn = {}
tau_samples_noburn = {}
phi_samples_noburn = {}
rho_samples_noburn = {}
gvar_samples_noburn = {}


for i in range(nchains):
    weights_samples_noburn[i] = weights_samples[i][:,:,int(N_samples/2):]
    c_samples_noburn[i] = c_samples[i][:,:,int(N_samples/2):]

    alpha_samples_noburn[i] = alpha_samples[i][int(N_samples/2):]
    sigma_samples_noburn[i] = sigma_samples[i][int(N_samples/2):]
    tau_samples_noburn[i] = tau_samples[i][int(N_samples/2):]
    phi_samples_noburn[i] = phi_samples[i][int(N_samples/2):]
    rho_samples_noburn[i] = rho_samples[i][int(N_samples/2):]
    gvar_samples_noburn[i] = gvar_samples[i][:,int(N_samples/2):]


objmcmc_noburn = {}
objmcmc_noburn['settings']=settings
objmcmc_noburn['samples'] = {}
objmcmc_noburn['samples']['w'] = weights_samples_noburn
objmcmc_noburn['samples']['alpha'] = alpha_samples_noburn
objmcmc_noburn['samples']['sigma'] = sigma_samples_noburn
objmcmc_noburn['samples']['tau'] = tau_samples_noburn
objmcmc_noburn['samples']['phi'] = phi_samples_noburn
objmcmc_noburn['samples']['rho'] = rho_samples_noburn
objmcmc_noburn['samples']['gvar'] = gvar_samples_noburn


folder_name_noburn = folder_name+'/noburn'
if not os.path.exists(folder_name_noburn):
    os.makedirs(folder_name_noburn)

# Posterior Predictive Degree plot

plot_degreepostpred_sparse_chains(Z, objmcmc_noburn, folder_name_noburn, ndraws=100)

# Plot the Weights
if settings['estimate_weights']:
    weights_samples_combined = weights_samples[0]
    for i in range(1, nchains):
        weights_samples_combined = np.concatenate((weights_samples_combined, weights_samples[i]), axis=-1)

    weights_samples_combined = weights_samples_noburn[0]
    for i in range(1, nchains):
        weights_samples_combined = np.concatenate((weights_samples_combined, weights_samples_noburn[i]), axis=-1)

    for t in range(T):
        plt.figure()
        for k in range(np.minimum(tw.shape[0], 50)):
            plt.plot([k, k], np.quantile(weights_samples_combined[indt[t][k],t,:],[.025,.975]), color='red',linewidth=3)
            plt.plot(k, tw[indt[t][k], t], marker='x', color='green', linewidth=2)

        plt.xlim(-1.1,np.minimum(Na, 50)+.5)
        # plt.legend('95# credible intervals', 'True value')
        plt.xlabel('Index of node (sorted by dec. degree)')
        plt.ylabel('Weights')
        plt.show()

        plt.savefig('{}/weights_high_degree_T_{}.png'.format(folder_name,t))

        plt.figure()
        for k in range(1,np.minimum(tw.shape[0], 101)):
            plt.plot([k, k], np.quantile(np.log(weights_samples_combined[indt[t][-k],t,:]),[.025,.975]), color='red',linewidth=3)
            plt.plot(k, np.log(tw[indt[t][-k], t]), marker='x', color='green', linewidth=2)

        plt.xlim(-1.1,np.minimum(Na, 100)+.5)
        # plt.legend('95# credible intervals', 'True value')
        plt.xlabel('Index of node (sorted by dec. degree)')
        plt.ylabel('Log Weights')
        plt.show()

        plt.savefig('{}/weights_low_degree_T_{}.png'.format(folder_name, t))


    plt.figure()
    for i in range(nchains):
        # plt.plot(np.arange(1, N_samples * thin + 1, thin), w_rate_samples[i][0,:])
        plt.plot(np.arange(1, N_samples * thin + 1, thin),
                np.divide(np.cumsum(w_rate_samples[i][0,:]), np.arange(1, N_samples + 1)), '-')

    # plt.legend('95# credible intervals', 'True value')
    plt.xlabel('MCMC iterations')
    plt.ylabel('w_rate')
    plt.show()

    plt.figure()
    for i in range(nchains):
        plt.plot(np.arange(1, N_samples * thin + 1, thin), epsilon_samples[i][0,:])
        # plt.plot(np.arange(1, N_samples * thin + 1, thin),
        #         np.divide(np.cumsum(epsilon_samples[i][0,:]), np.arange(1, N_samples + 1)), '-')

    # plt.legend('95# credible intervals', 'True value')
    plt.xlabel('MCMC iterations')
    plt.ylabel('epsilon')

    tremaining_mass = np.sum(w[:,~tindchain],1)
    remaining_mass_samples = {}
    for i in range(nchains):
        remaining_mass_samples[i] = np.sum(weights_samples[i][~tindchain_trunc,:,:],0)



    for t in range(T):
        plt.figure()
        for i in range(nchains):
            plt.plot(np.arange(1, N_samples * thin + 1, thin), remaining_mass_samples[i][t,:])
            plt.plot(np.arange(1, N_samples * thin + 1, thin), tremaining_mass[t] * np.ones(N_samples), 'g--', linewidth=3)
            # plt.plot(np.arange(1, N_samples * thin + 1, thin),
            #          np.divide(np.cumsum(remaining_mass_samples[i][t,:]), np.arange(1, N_samples + 1)), '-')
        # plt.legend('95# credible intervals', 'True value')
        plt.xlabel('MCMC iterations')
        plt.ylabel('Remaining Mass')
        plt.show()

        plt.savefig('{}/remaining_mass_T_{}.png'.format(folder_name,t))


# Plot the Counts
if settings['estimate_C']:
    c_samples_combined = c_samples[0]
    for i in range(1, nchains):
        c_samples_combined = np.concatenate((c_samples_combined, c_samples[i]), axis=-1)

    c_samples_combined = c_samples_noburn[0]
    for i in range(1, nchains):
        c_samples_combined = np.concatenate((c_samples_combined, c_samples_noburn[i]), axis=-1)
    for t in range(T):
        plt.figure()
        for k in range(np.minimum(tw.shape[0], 50)):
            plt.plot([k, k], np.quantile(c_samples_combined[indt[t][k],t,:],[.025,.975]), color='red',linewidth=3)
            plt.plot(k, tc[indt[t][k], t], marker='x', color='green', linewidth=2)

        plt.xlim(-1.1,np.minimum(Na, 50)+.5)
        # plt.legend('95# credible intervals', 'True value')
        plt.xlabel('Index of node (sorted by dec. degree)')
        plt.ylabel('C')
        plt.show()

        plt.savefig('{}/C_T_{}.png'.format(folder_name, t))

# Plot logposterior
if settings['logposterior']:
    for j in range(1):
        plt.figure()
        for i in range(nchains):
            plt.plot(np.arange(1,N_samples*thin+1,thin), logpost_samples[i][j,:])
        if method == 'BFRY':
            tu = bfry_sample_u(tw, talpha, tsigma, L, settings)
            logpost_true = bfry_log_posterior(tw, tc, tu, tm, tn_new, tn_old, talpha, tsigma, ttau, tphi, trho, tgvar,
                                         settings)
            plt.plot(np.arange(1,N_samples*thin+1,thin), logpost_true[0]*np.ones(N_samples), 'g--',linewidth=3)

        plt.xlabel('MCMC iterations')
        plt.ylabel('Log Posterior')
        plt.show()
        plt.locator_params(axis='x', nbins=4)
        plt.tight_layout()

        plt.savefig('{}/logpost_{}.png'.format(folder_name,j))


# Plot alpha


if settings['estimate_alpha']:
    plt.figure()
    for i in range(nchains):
        plt.plot(np.arange(1,N_samples*thin+1,thin), alpha_samples[i])
        # plt.plot(np.arange(1,N_samples*thin+1,thin), np.divide(np.cumsum(alpha_samples[i]),np.arange(1,N_samples*thin+1,thin)), '-')

    plt.plot(np.arange(1,N_samples*thin+1,thin), talpha*np.ones(N_samples), 'g--',linewidth=3)

    # plt.legend('95# credible intervals', 'True value')
    plt.xlabel('MCMC iterations')
    plt.ylabel(r'$\alpha$')
    plt.show()
    plt.locator_params(axis='x', nbins=4)
    plt.tight_layout()

    plt.savefig('{}/alpha.png'.format(folder_name))
    #xlim([0, N_Gibbs.*(log(wsnew) -log(Wst))])


if settings['estimate_sigma']:
    plt.figure()
    for i in range(nchains):
        plt.plot(np.arange(1, N_samples*thin + 1, thin), sigma_samples[i])
        # plt.plot(np.arange(1, N_samples * thin + 1, thin),
        #          np.divide(np.cumsum(sigma_samples[i]), np.arange(1, N_samples * thin + 1, thin)),'-')
    plt.plot(np.arange(1, N_samples*thin + 1, thin), tsigma * np.ones(N_samples), 'g--', linewidth=3)


    # plt.legend('95# credible intervals', 'True value')
    plt.xlabel('MCMC iterations')
    plt.ylabel(r'$\sigma$')
    plt.show()
    plt.locator_params(axis='x', nbins=4)
    plt.tight_layout()

    plt.savefig('{}/sigma.png'.format(folder_name))

if settings['estimate_tau']:
    plt.figure()
    for i in range(nchains):
        plt.plot(np.arange(1, N_samples*thin + 1, thin), tau_samples[i])
        # plt.plot(np.arange(1, N_samples * thin + 1, thin),
        #          np.divide(np.cumsum(tau_samples[i]), np.arange(1, N_samples * thin + 1, thin)),'-')

    plt.plot(np.arange(1, N_samples*thin + 1, thin), ttau * np.ones(N_samples), 'g--', linewidth=3)

    # plt.legend('95# credible intervals', 'True value')
    plt.xlabel('MCMC iterations')
    plt.ylabel(r'$\tau$')
    plt.show()
    plt.locator_params(axis='x', nbins=4)
    plt.tight_layout()

    plt.savefig('{}/tau.png'.format(folder_name))


if settings['estimate_phi']:
    plt.figure()
    for i in range(nchains):
        plt.plot(np.arange(1, N_samples*thin + 1, thin), phi_samples[i])
        # plt.plot(np.arange(1, N_samples*thin + 1, thin),
        #          np.divide(np.cumsum(phi_samples[i]), np.arange(1, N_samples*thin + 1, thin)),'-')
    plt.plot(np.arange(1, N_samples*thin + 1, thin), tphi * np.ones(N_samples), 'g--', linewidth=3)


    # plt.legend('95# credible intervals', 'True value')
    plt.xlabel('MCMC iterations')
    plt.ylabel(r'$\phi$')
    plt.show()
    plt.locator_params(axis='x', nbins=4)
    plt.tight_layout()

    plt.savefig('{}/phi.png'.format(folder_name))

if settings['gcontrol']:
    for t in range(T):

        plt.figure()
        for i in range(nchains):
            plt.plot(np.arange(1, N_samples * thin + 1, thin), gvar_samples[i][t,:])
            # plt.plot(np.arange(1, N_samples * thin + 1, thin),
            #          np.divide(np.cumsum(gvar_samples[i][t,:]), np.arange(1, N_samples * thin + 1, thin)), '-')

        plt.plot(np.arange(1, N_samples * thin + 1, thin), tgvar[t] * np.ones(N_samples), 'g--', linewidth=3)

        # plt.legend('95# credible intervals', 'True value')
        plt.xlabel('MCMC iterations')
        plt.ylabel('gvar')
        plt.show()

        plt.savefig('{}/gvar_{}.png'.format(folder_name,t))

if settings['estimate_hypers']:

    plt.figure()
    for i in range(nchains):

    # plt.plot(np.arang   e(1, N_samples * thin + 1, thin), hyper_rate_samples)
        plt.plot(np.arange(1, N_samples * thin + 1, thin),
                 np.divide(np.cumsum(hyper_rate_samples[i]), np.arange(1, N_samples * thin + 1, thin)), '-')

    # plt.legend('95# credible intervals', 'True value')
    plt.xlabel('MCMC iterations')
    plt.ylabel('hyper rate')
    plt.show()

if settings['estimate_rho']:
    plt.figure()
    for i in range(nchains):
        plt.plot(np.arange(1, N_samples*thin + 1, thin), rho_samples[i])
        # plt.plot(np.arange(1, N_samples*thin + 1, thin), np.divide(np.cumsum(rho_samples[i]),
        #                                                            np.arange(1, N_samples*thin + 1, thin)), '-')

    plt.plot(np.arange(1, N_samples*thin + 1, thin), trho * np.ones(N_samples), 'g--', linewidth=3)

    # plt.legend('95# credible intervals', 'True value')
    plt.xlabel('MCMC iterations')
    plt.ylabel(r'$\rho$')
    plt.show()

    plt.savefig('{}/rho.png'.format(folder_name))

stats_noburn = {}
stats_noburn['weights_mean'] = np.zeros((K, T))
stats_noburn['weights_std'] = np.zeros((K, T))
stats_noburn['weights_05'] = np.zeros((K, T))
stats_noburn['weights_95'] = np.zeros((K, T))


for k in range(T):
    stats_noburn['weights_mean'][:, k] = np.mean(weights_samples_combined[:,k,:], 1)
    stats_noburn['weights_std'][:, k] = np.std(weights_samples_combined[:,k,:], 1)
    stats_noburn['weights_05'][:, k] = np.quantile(weights_samples_combined[:,k,:], .05, 1)
    stats_noburn['weights_95'][:, k] = np.quantile(weights_samples_combined[:,k,:], .95, 1)


deg_all = np.squeeze(np.asarray(np.sum(Z[-1],0)))
ind = np.argsort(deg_all)
ind = ind[::-1]
indmax =5

plt.figure()
for i in range(indmax):
    if i==0:
        plt.fill_between(np.arange(T), stats_noburn['weights_95'][ind[i], :], stats_noburn['weights_05'][ind[i], :],
                         alpha=0.2, label="95% credible interval")
        plt.plot(tw[ind[i], :], '--', label="True Weights")
    else:
        plt.fill_between(np.arange(T), stats_noburn['weights_95'][ind[i],:],stats_noburn['weights_05'][ind[i],:],alpha=0.2)
    # plt.plot(stats_noburn['weights_mean'][ind[i], :])

        plt.plot(tw[ind[i], :], '--')

plt.ylim([0.4,3.6])
plt.xlabel('Time')
plt.ylabel('Weights')
plt.legend()
plt.tight_layout()

plt.savefig('{}/weights_top_5.png'.format(folder_name))