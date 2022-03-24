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
tphi = 10             # tunes dependence in dependent gamma process
T = 4                        # Number of time steps
trho = 0                    # death rate for latent interactions

settings = {}

settings['dt']=1

# Choose truncation level L

settings['L']=10000
settings['fromggprnd']=1
settings['nchains'] = 1
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

[Z, w, c, KT,  N_new, N_old, Nall, M, indchain, indcnz]= ggp_dyngraphrnd_sparse(method, talpha, tsigma, ttau, T, tphi,
                                                                                trho, tgvar, settings)

f = open('store_simulated.pckl', 'wb')
pickle.dump([Z, w, c, KT,  N_new, N_old, Nall, M, indchain, indcnz], f)
f.close()



print(Z[0].shape)
print(sum(M))
print(sum(indchain))
print([np.sum(np.sum(z,1)>0) for z in Z])

color=plt.cm.rainbow(np.linspace(0,1,T))
for t in range(T):
    plot_degree(np.squeeze(Z[t].todense()),color=color[t])
    plt.legend(('t=0','t=1','t=2','t=3'))

[T, K] = w.shape

tindchain = indchain.copy()
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

if settings['gcontrol']:
    gvar = np.random.gamma(settings['g_a'], 1 / settings['g_b'],T)
    gvar = tgvar.copy()
else:
    gvar = tgvar.copy()



if settings['estimate_ncounts'] or settings['gcontrol']:

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


    if settings['typegraph'] == 'simple':
        issimple = True
    else:
        issimple = False

    ind = [None] * T
    trZ = [None] * T
    linidxs = [None] * T
    for t in range(T):
        G = Z[t]

        if issimple:  # If no self-loops
            G2 = triu((G + np.transpose(G)) > 0, 1)  # G2 upper triangular

        else:
            G2 = triu((G + np.transpose(G)) > 0)
        linidxs[t] = np.nonzero(G2)
        ind[t] = np.nonzero(G2)  # the indices refer to the upper triangle

        trZ[t] = G2

if settings['estimate_rho'] or settings['gcontrol'] or settings['logposterior']:
    tn_new = [None] * T
    tn_old = [None] * T
    for t in range(T):
        # make it symmetric
        N_new_t = N_new[t][indchain, :][:, indchain]

        temp = N_new_t + np.transpose(N_new_t)

        # make it upper triangular for the sampler to work correctly
        tn_new[t] = triu(temp, 1)

        N_old_t = N_old[t][indchain, :][:, indchain]

        tempo = N_old_t + np.transpose(N_old_t)
        tn_old[t] = triu(tempo, 1)

######################################################
# Initialise structures for each chain
######################################################

# Set up separate chains
nchains = settings['nchains']

weights_samples = {}
c_samples = {}
logpost_samples = {}
alpha_samples = {}
sigma_samples = {}
tau_samples = {}
phi_samples = {}
rho_samples = {}
gvar_samples = {}
w_rate_samples = {}
n_rate_samples = {}
hyper_rate_samples = {}
epsilon_samples = {}

mnnew_st = {}
mnold_st = {}
mnall_st = {}


for i in range(nchains):
    weights_samples[i] = np.zeros((K, T, N_samples), dtype='float')
    c_samples[i] = np.zeros((K, T, N_samples), dtype='int')
    logpost_samples[i] = np.zeros((7,N_samples))
    alpha_samples[i] = np.zeros(N_samples)
    sigma_samples[i] = np.zeros(N_samples)
    tau_samples[i] = np.zeros(N_samples)
    phi_samples[i] = np.zeros(N_samples)
    rho_samples[i] = np.zeros(N_samples)
    gvar_samples[i] = np.zeros((T, N_samples))
    w_rate_samples[i] = np.zeros((T, N_samples))
    n_rate_samples[i] = np.zeros((T, N_samples))
    hyper_rate_samples[i] = np.zeros(N_samples)
    epsilon_samples[i] = np.zeros((T, N_samples))


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

    if settings['estimate_alpha']:
        alpha = np.random.uniform(high=talpha)

    else:
        alpha = talpha

    if settings['estimate_sigma']:
        sigma=np.random.uniform()

    else:
        sigma = tsigma

    if settings['estimate_tau']:
        tau = np.random.uniform()
    else:
        tau = ttau

    if settings['estimate_phi']:
        phi = np.random.uniform()
    else:
        phi = tphi

    if settings['estimate_rho']:
        rho = np.random.gamma(settings['hyper_rho'][0], 1 / settings['hyper_rho'][1])

    else:
        rho = trho

    if settings['estimate_weights']:
        weights = np.random.uniform(size=(tw.shape))  # random initialization of the weights

    else:
        weights = tw.copy()

    if settings['estimate_C']:
        # initialise counts
        if settings['estimate_weights']:
            counts = np.random.poisson(phi * weights)
        else:
            counts = np.random.poisson(phi * weights) + 100

    else:
        counts = tc.copy()

    if settings['estimate_ncounts']:
        # Initialize interaction counts
        Nnew = [None] * T
        Nold = [None] * T
        Ntot = [None] * T

        dt = settings['dt']
        k = tw.shape[0]

        for t in range(T):
            logw = np.log(weights[:, t]).copy()
            Nnew[t] = csr_matrix((k, k), dtype='int')
            iid1 = ind[t][0]
            iid2 = ind[t][1]
            lograte_poi = np.log(2) + np.log(gvar[t]) + logw[iid1] + logw[iid2]
            lograte_poi[iid1 == iid2] = np.log(gvar[t]) + 2 * logw[iid1[iid1 == iid2]]

            Nnew[t][linidxs[t]] = np.random.poisson(np.exp(lograte_poi)) * 2

            Nold[t] = csr_matrix((k, k), dtype='int')
            if t > 0:
                Nold[t][linidxs[t]] = np.random.binomial(Ntot[t - 1][linidxs[t]], np.exp(-rho * dt))

            Ntot[t] = Nnew[t] + Nold[t]

    if settings['gcontrol']:
        gvar = np.random.gamma(settings['g_a'], 1 / settings['g_b'])
    else:
        gvar = np.ones(T, dtype='int')

    mnnew = np.zeros((K, T), dtype='int')
    mnold = np.zeros((K, T), dtype='int')
    mnall = np.zeros((K, T), dtype='int')

    rate_hyper=np.zeros(N_Gibbs)
    n_rate = np.zeros(T)
    rate = np.zeros((T, N_Gibbs))
    epsilon = settings['leapfrog']['epsilon'] / (K - 1) ** (1 / 4) * np.ones(T)  # Leapfrog stepsize

    print('***************\n')
    print('Starting Chain {}\n'.format(n))
    print('***************\n')

    [kk ,T] =tw.shape

    for i in range(N_Gibbs):
        if np.mod(i, 100) == 0:
            print(i)
        if np.mod(i, 1000) == 0:
            print('sigma = {}'.format(sigma))
            print('alpha = {}'.format(alpha))
            print('tau = {}'.format(tau))
            print('phi = {}'.format(phi))


        if settings['estimate_ncounts']:
            if (np.remainder(i, settings['nlatent']) == 0):
                i_ncounts = int((i/ settings['nlatent']))

                # Sample new interaction counts every nlatent steps
                n_rate = np.zeros(T)
                for pp in range(T):
                    id = ind[pp]
                    ind1 = id[0]
                    ind2 = id[1]
                    logw = np.log(weights[:, pp])
                    [new_inter, old_inter, Mn,n_rate[pp]] = bfry_sample_interaction_counts(pp, trZ[pp], logw, Nnew, Nold, ind1, ind2, T,
                                                                               gvar[pp], rho, settings)

                    Nnew[pp] = new_inter  # upper triangular (full info)
                    Nold[pp] = old_inter  # upper triangular (full info)
                    Ntot[pp] = new_inter + old_inter  # upper triangular (full info)

                    mnnew[:, pp] = Mn  # matrix mm should be of size K x T

                    mnold[:, pp] = np.transpose(np.squeeze(np.asarray(np.sum(Nold[pp], 0)))) \
                                   + np.squeeze(np.asarray(np.sum(Nold[pp], 1)))  # - diag(squeeze(N_new(t, :, :)))
                    mnall[:, pp] = np.transpose(np.squeeze(np.asarray(np.sum(Ntot[pp], 0)))) \
                                   + np.squeeze(np.asarray(np.sum(Ntot[pp], 1)))


        else:
            mnall = tm
            mnnew = tm

        if settings['estimate_weights']:

            L = kk

            u = bfry_sample_u(weights, alpha, sigma, L, settings)

            [weights, rate[:, i]] = bfry_sample_weights(weights, counts, u, mnnew, epsilon, alpha, sigma, tau, L, phi, gvar, settings)

            if (i < settings['leapfrog']['nadapt']) & (i>0):  # Adapt the stepsize
                adapt_lag = 500
                epsilon = np.exp(np.log(epsilon) + .01 * (np.mean(rate[:,max(0,i-adapt_lag):i], 1) - 0.65))




        if settings['estimate_C']:
            L = kk
            [counts,accept_counts] = ggp_sample_C(counts, weights, phi, alpha, sigma, tau, L)

        if settings['estimate_hypers']:

            [alpha, sigma, tau, phi, rate_hyper[i]] = ggp_sample_hyperparameters(weights, counts, alpha, sigma, tau, phi, settings)

            if (i < settings['rw_std'][4]) & (i>0):  # Adapt the stepsize
                adapt_lag = 500
                settings['rw_std'][0] = np.exp(np.log(settings['rw_std'][0]) + .01 * (np.mean(rate_hyper[max(0,i-adapt_lag):i]) - 0.23))
                settings['rw_std'][1] = np.exp(np.log(settings['rw_std'][1]) + .01 * (np.mean(rate_hyper[max(0,i-adapt_lag):i]) - 0.23))
                settings['rw_std'][2] = np.exp(np.log(settings['rw_std'][2]) + .01 * (np.mean(rate_hyper[max(0,i-adapt_lag):i]) - 0.23))
                settings['rw_std'][3] = np.exp(np.log(settings['rw_std'][3]) + .01 * (np.mean(rate_hyper[max(0,i-adapt_lag):i]) - 0.23))


        if settings['gcontrol']:
            if i >= 0:
                if settings['estimate_ncounts']:
                    if (np.remainder(i, settings['ngvar']) == 0):
                        n_new = [None] * T
                        for t in range(T):
                            # make it symmetric
                            N_new_t = Nnew[t][indchain_trunc, :][:, indchain_trunc]

                            temp = N_new_t + np.transpose(N_new_t)

                            # make it upper triangular for the sampler to work correctly
                            n_new[t] = triu(temp, 1)

                        gvar = ggp_sample_gvar_mh(weights, n_new, gvar, settings['g_a'], settings['g_b'])

                else:
                    if (np.remainder(i, settings['ngvar']) == 0):
                        gvar = ggp_sample_gvar_mh(weights, tn_new, gvar, settings['g_a'], settings['g_b'])


        if settings['estimate_rho']:
            if settings['estimate_ncounts']:
                if (np.remainder(i, settings['nlatent']) == 0) :
                    n_new = [None] * T
                    n_old = [None] * T
                    for t in range(T):
                        # make it symmetric
                        N_new_t = Nnew[t][indchain_trunc, :][:, indchain_trunc]

                        temp = N_new_t + np.transpose(N_new_t)

                        # make it upper triangular for the sampler to work correctly
                        n_new[t] = triu(temp, 1)

                        N_old_t = Nold[t][indchain_trunc, :][:, indchain_trunc]

                        tempo = N_old_t + np.transpose(N_old_t)
                        n_old[t] = triu(tempo, 1)

                    rho = ggp_sample_rho(rho, n_old, n_new, settings)

            else:
                rho = ggp_sample_rho(rho, tn_old, tn_new, settings)

        # Plot log-posterior
        logpost = np.zeros(7)
        if settings['logposterior']:
            if settings['estimate_ncounts']:

                n_new = [None] * T
                n_old = [None] * T
                for t in range(T):
                    # make it symmetric
                    N_new_t = Nnew[t][indchain_trunc, :][:, indchain_trunc]

                    temp = N_new_t + np.transpose(N_new_t)

                    # make it upper triangular for the sampler to work correctly
                    n_new[t] = triu(temp, 1)

                    N_old_t = Nold[t][indchain_trunc, :][:, indchain_trunc]

                    tempo = N_old_t + np.transpose(N_old_t)
                    n_old[t] = triu(tempo, 1)

                logpost = bfry_log_posterior(weights, counts, u, mnnew, n_new, n_old, alpha, sigma, tau, phi, rho, gvar, settings)

            else:
                logpost = bfry_log_posterior(weights, counts, u, mnnew, tn_new, tn_old, alpha, sigma, tau, phi, rho, gvar, settings)


        if (i >= N_burn and np.remainder((i - N_burn), thin) == 0):
            indd = int(((i - N_burn) / thin))

            weights_samples[n][:, :, indd] = weights
            c_samples[n][:, :, indd] = counts
            for j in range(7):
                logpost_samples[n][j,indd] = logpost[j]
            alpha_samples[n][indd] = alpha
            sigma_samples[n][indd] = sigma
            tau_samples[n][indd] = tau
            phi_samples[n][indd] = phi
            rho_samples[n][indd] = rho
            gvar_samples[n][:,indd] = gvar
            w_rate_samples[n][:,indd] = rate[:,i]
            hyper_rate_samples[n][indd] = rate_hyper[i]
            epsilon_samples[n][:,indd] = epsilon
            n_rate_samples[n][:,indd] = n_rate

        if n == 0:
            if i == 99:
                t1 = (time.time() - t0) * N_Gibbs * nchains / (3600 * 100)
                print('Estimated computation time: {} hours\n'.format(t1))
                # print('Estimated end of computation: %s \n', datestr(now + toc * N_Gibbs/3600/24))
                print('***************\n')



t1 = time.time()

print('***************\n')
print('END dynamic model\n')
print('***************\n')
print('Computation time = {} hours\n'.format((t1-t0) / 3600))
print('***************\n')

for i in range(nchains):
    print(np.mean(hyper_rate_samples[i]))

##################################################
# Save results for combining chains
##################################################

f = open('store_simulated_results_chain_1.pckl', 'wb')
pickle.dump([settings, weights_samples, c_samples, logpost_samples, phi_samples, sigma_samples,
             alpha_samples, tau_samples, rho_samples, rate, epsilon, rate_hyper,
             gvar_samples, w_rate_samples, hyper_rate_samples, epsilon_samples], f)
f.close()


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

# # Plot the Weights
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
        # for i in range(nchains):
        #     plt.plot(np.arange(1,N_samples*thin+1,thin), np.divide(np.cumsum(logpost_samples[i]),np.arange(1,N_samples*thin+1,thin)), 'g-')

        # plt.legend('95# credible intervals', 'True value')
        plt.xlabel('MCMC iterations')
        plt.ylabel('Log Posterior')
        plt.tight_layout()

        plt.show()

        plt.savefig('{}/logpost_{}.png'.format(folder_name,j))
        #xlim([0, N_Gibbs.*(log(wsnew) -log(Wst))])


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


    plt.plot(np.arange(1, N_samples*thin + 1, thin), trho * np.ones(N_samples), 'g--', linewidth=3)

    # plt.legend('95# credible intervals', 'True value')
    plt.xlabel('MCMC iterations')
    plt.ylabel(r'$\rho$')
    plt.show()

    plt.savefig('{}/rho.png'.format(folder_name))
