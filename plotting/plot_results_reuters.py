import numpy as np
import pickle
import random
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
from plot_degree_sparse import plot_degree_sparse
plt.rcParams.update({'font.size':16})

f = open('store_reuters.pckl', 'rb')
tG, G_all, timestamps, G, Nnew, words = pickle.load(f)
f.close()


f = open('store_reuters_results_all.pckl', 'rb')
settings, stats, stats_noburn, phi_st, sigma_st, alpha_st, tau_st = pickle.load(f)
f.close()

folder_name = 'reuters'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

G_all = csr_matrix.sign(G_all)

T = len(tG)
degs = np.sum(G_all,0)


nchains = settings['nchains']
N_samples = settings['N_samples']
thin = settings['thin']

plt.figure()
for i in range(nchains):
    # plt.plot(phi_st[i])
    plt.plot(np.arange(1, N_samples * thin + 1, thin), phi_st[i])
plt.xlabel('MCMC samples')
plt.ylabel(r'$\phi$')
plt.savefig('{}/phi.png'.format(folder_name))

plt.figure()
for i in range(nchains):
    # plt.plot(alpha_st[i])
    plt.plot(np.arange(1, N_samples * thin + 1, thin), alpha_st[i])
plt.xlabel('MCMC samples')
plt.ylabel(r'$\alpha$')
plt.savefig('{}/alpha.png'.format(folder_name))

plt.figure()
for i in range(nchains):
    # plt.plot(sigma_st[i])
    plt.plot(np.arange(1, N_samples * thin + 1, thin), sigma_st[i])
plt.xlabel('MCMC samples')
plt.ylabel(r'$\sigma$')
plt.savefig('{}/sigma.png'.format(folder_name))

plt.figure()
for i in range(nchains):
    # plt.plot(tau_st[i])
    plt.plot(np.arange(1, N_samples * thin + 1, thin), tau_st[i])
plt.xlabel('MCMC samples')
plt.ylabel(r'$\tau$')
plt.savefig('{}/tau.png'.format(folder_name))


w_mean = stats['weights_mean']


plt.figure()
plt.plot(np.sum(w_mean, 0))

plt.figure()
plt.plot(w_mean[-1,:])

plt.figure()
plt.plot(stats['weights_95'][0,:])
plt.plot(stats['weights_05'][0,:])

deg_all = np.squeeze(np.asarray(np.sum(G_all,0)))
ind = np.argsort(deg_all)
ind = ind[::-1]
indmax =5
deg_all[ind[:indmax]]

K = G_all.shape[0]
deg = np.zeros((K,T),dtype='int')
for t in range(T):
    deg[:,t] = np.squeeze(np.asarray(np.sum(G[t],0)))


plt.figure()
for i in range(indmax):
    plt.fill_between(np.arange(T), stats['weights_95'][ind[i],:],stats['weights_05'][ind[i],:],alpha=0.2)
    plt.plot(stats['weights_mean'][ind[i], :])

plt.xlabel('Week')
plt.ylabel('Weights')
plt.tight_layout()

plt.savefig('{}/weights_top_5.png'.format(folder_name))

# xlim([1, T])
# xlabel('Time')
# ylabel('Sociability weights')
# box off
# saveas(gca, 'facebookweights', 'png')


plt.figure()
for i in range(indmax):
    plt.plot(deg[ind[i], :])

plt.xlabel('Week')
plt.ylabel('Degree')
plt.tight_layout()

plt.savefig('{}/degrees_top_5.png'.format(folder_name))

plt.figure()
h1 = plt.fill_between(np.arange(T), np.sum(stats['weights_95'], 0),np.sum(stats['weights_05'], 0),alpha = 0.2)
h2 = plt.plot(np.sum(stats['weights_mean'],0))

plt.xlabel('Week')
plt.ylabel('Mean Weight')
plt.tight_layout()

plt.savefig('{}/weights_mean.png'.format(folder_name))


wordsplot_all = {0:['plane', 'attack'], 1:['al_quaeda', 'taliban', 'bin_laden'],2:['anthrax', 'letter']}
wordsplot_all2 = {0:['plane', 'attack'], 1:['al_quaeda', 'taliban', 'bin_laden'],2:['anthrax', 'letter']}
# Plot number of interactions at each time
for k in range(len(wordsplot_all)):
    wordsplot = wordsplot_all[k]
    plt.figure()
     # wordsplot = {'plane', 'attack'}
    for i in range(len(wordsplot)):
        indword = [wordsplot[i] == w for w in words.values()]
        plt.plot(np.transpose(deg[indword, :]))

        plt.xlabel('Weeks after 9/11')
        plt.ylabel('Number of Interactions')
        plt.tight_layout()

        plt.legend(wordsplot)

    plt.xlim([0,T-1])
    # plt.ylim([0,500])
    plt.savefig('{}/words_degree_{}.png'.format(folder_name,k))

for k in range(len(wordsplot_all)):
    wordsplot = wordsplot_all[k]
    plt.figure()
     # wordsplot = {'plane', 'attack'}
    for i in range(len(wordsplot)):
        indword = [wordsplot[i] == w for w in words.values()]
        all_inds = np.full(len(stats['weights_95']), False)
        all_inds[:len(indword)] = indword
        plt.fill_between(np.arange(T), np.squeeze(stats['weights_95'][all_inds, :]), np.squeeze(stats['weights_05'][all_inds, :]), alpha=0.2)
        plt.plot(np.squeeze(stats['weights_mean'][all_inds, :]))

        plt.xlabel('Weeks after 9/11')
        plt.ylabel('Weights')
        plt.tight_layout()

        plt.xlim([0, T - 1])
        # plt.ylim([0, 2])
    plt.legend(wordsplot)
    plt.savefig('{}/words_weights_{}.png'.format(folder_name,k))


folder_name_noburn = folder_name+'/noburn'
if not os.path.exists(folder_name_noburn):
    os.makedirs(folder_name_noburn)

plt.figure()
for i in range(indmax):
    plt.fill_between(np.arange(T), stats_noburn['weights_95'][ind[i],:],stats_noburn['weights_05'][ind[i],:],alpha=0.2)
    plt.plot(stats_noburn['weights_mean'][ind[i], :])

plt.xlabel('Week')
plt.ylabel('Weights')
plt.tight_layout()

plt.savefig('{}/weights_top_5.png'.format(folder_name_noburn))

plt.figure()
h1 = plt.fill_between(np.arange(T), np.sum(stats_noburn['weights_95'], 0),np.sum(stats_noburn['weights_05'], 0),alpha = 0.2)
h2 = plt.plot(np.sum(stats_noburn['weights_mean'],0))

plt.xlabel('Week')
plt.ylabel('Mean Weight')
plt.tight_layout()

plt.savefig('{}/weights_mean.png'.format(folder_name_noburn))

for k in range(len(wordsplot_all)):
    wordsplot = wordsplot_all[k]
    plt.figure()
     # wordsplot = {'plane', 'attack'}
    for i in range(len(wordsplot)):
        indword = [wordsplot[i] == w for w in words.values()]
        all_inds = np.full(len(stats_noburn['weights_95']), False)
        all_inds[:len(indword)] = indword
        plt.fill_between(np.arange(T), np.squeeze(stats_noburn['weights_95'][all_inds, :]), np.squeeze(stats_noburn['weights_05'][all_inds, :]), alpha=0.2)
        plt.plot(np.squeeze(stats_noburn['weights_mean'][all_inds, :]))

        plt.xlabel('Weeks after 9/11')
        plt.ylabel(r'Weights')
        plt.tight_layout()

        plt.xlim([0, T - 1])
        # plt.ylim([0, 2])
    plt.legend(wordsplot)
    plt.savefig('{}/words_weights_{}.png'.format(folder_name_noburn,k))

