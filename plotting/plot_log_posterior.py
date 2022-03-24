import numpy as np
import random
import matplotlib.pyplot as plt
import time
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../experiments'))
sys.path.insert(1, os.path.join(sys.path[0], '../data'))
sys.path.insert(1, os.path.join(sys.path[0], '../util'))

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

# np.warnings.filterwarnings('ignore')

# Folder to save results to

folder_name = 'parameters/logposterior'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Set the seed
random.seed(1001)

talpha = 100
tsigma = 0.6
ttau = 1 # Parameters gamma process
tphi = 400          # tunes dependence in dependent gamma process
T = 12                       # Number of time steps
trho = 0.1                     # death rate for latent interactions

settings = {}

settings['dt']=1

# Choose truncation level L

settings['L']=100000
settings['fromggprnd']=0
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


N_Gibbs = 5000
N_burn = 0*N_Gibbs
thin = 25
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


print(Z[0].shape)
print(sum(M))
print(sum(indchain))
print([np.sum(np.sum(z,1)>0) for z in Z])


[T, K] = w.shape

L = settings['L']



for t in range(T):

    csr_matrix.resize(Z[t], (L, L))
    csr_matrix.resize(N_new[t], (L, L))
    csr_matrix.resize(N_old[t], (L, L))
    csr_matrix.resize(Nall[t], (L, L))



w_new = np.zeros((T,L),dtype=float)
c_new = np.zeros((T,L),dtype=int)
M_new = np.zeros((L,T),dtype=int)

w_new[:,:w.shape[1]] = w
c_new[:,:c.shape[1]] = c
M_new[:w.shape[1],:] = M



indchain_new = np.full(L, False)
indchain_new[:len(indchain)] = indchain


w = w_new
c = c_new
M = M_new

indchain = indchain_new



indlog = np.full(len(indchain), False)
indlog[indchain] = True


tindchain = indchain.copy()
indchain = np.full(len(indchain), True)

indlog = np.full(len(indchain), False)
indlog[indchain] = True


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


# Plot logposterior for changing values of different hyperparameters:

# Sigma

sigma_all = np.linspace(0.05,0.95,100)
logpost_true = np.zeros(len(sigma_all))

for i in range(len(sigma_all)):
    tu = bfry_sample_u(tw, talpha, tsigma, L, settings)
    logpost_all = bfry_log_posterior(tw, tc, tu, tm, tn_new, tn_old, talpha, sigma_all[i], ttau, tphi, trho, tgvar,
                                 settings)

    logpost_true[i] = logpost_all[0]

plt.figure()

plt.plot(sigma_all,logpost_true)

plt.axvline(x=tsigma)
plt.xlabel(r'$\sigma$')
plt.ylabel('Log Posterior')
plt.show()

plt.savefig('{}/logpost_sigma.png'.format(folder_name))


# Alpha

alpha_all = np.linspace(10,1000,100)
logpost_true = np.zeros(len(alpha_all))

for i in range(len(sigma_all)):
    tu = bfry_sample_u(tw, talpha, tsigma, L, settings)
    logpost_all = bfry_log_posterior(tw, tc, tu, tm, tn_new, tn_old, alpha_all[i], tsigma, ttau, tphi, trho, tgvar,
                                 settings)

    logpost_true[i] = logpost_all[0]

plt.figure()

plt.plot(alpha_all,logpost_true)

plt.axvline(x=talpha)
plt.xlabel(r'$\alpha$')
plt.ylabel('Log Posterior')
plt.show()

plt.savefig('{}/logpost_alpha.png'.format(folder_name))

# Sigma

tau_all = np.linspace(0.1,5,100)
logpost_true = np.zeros(len(tau_all))

for i in range(len(sigma_all)):
    tu = bfry_sample_u(tw, talpha, tsigma, L, settings)
    logpost_all = bfry_log_posterior(tw, tc, tu, tm, tn_new, tn_old, talpha, tsigma, tau_all[i], tphi, trho, tgvar,
                                 settings)

    logpost_true[i] = logpost_all[0]

plt.figure()

plt.plot(tau_all,logpost_true)

plt.axvline(x=ttau)
plt.xlabel(r'$\tau$')
plt.ylabel('Log Posterior')
plt.show()

plt.savefig('{}/logpost_tau.png'.format(folder_name))

# Sigma

phi_all = np.linspace(10,5000,100)
logpost_true = np.zeros(len(phi_all))

for i in range(len(sigma_all)):
    tu = bfry_sample_u(tw, talpha, tsigma, L, settings)
    logpost_all = bfry_log_posterior(tw, tc, tu, tm, tn_new, tn_old, talpha, tsigma, ttau, phi_all[i], trho, tgvar,
                                 settings)

    logpost_true[i] = logpost_all[0]

plt.figure()

plt.plot(phi_all,logpost_true)

plt.axvline(x=tphi)
plt.xlabel(r'$\phi$')
plt.ylabel('Log Posterior')
plt.show()

plt.savefig('{}/logpost_phi.png'.format(folder_name))

