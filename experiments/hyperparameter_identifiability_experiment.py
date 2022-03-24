import numpy as np
import random
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
from parse import csr_matrix, triu, diags, lil_matrix, coo_matrix
from plot_degree_sparse import plot_degree_sparse
from plot_degree_post_pred_sparse import *
from bfry_log_posterior import *

plt.rcParams.update({'font.size':16})


#######################################
# Sigma
#######################################

# Folder to save results to

folder_name = 'parameters/sigma'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Set the seed
random.seed(1001)

talpha = 200
ttau = 1 # Parameters gamma process
tphi =10                      # tunes dependence in dependent gamma process
T = 4                        # Number of time steps
trho = 0.1                     # death rate for latent interactions
tgvar = np.ones(T,dtype='int')

settings = {}

settings['dt']=1


settings['L']=20000
settings['fromggprnd']=1
settings['onlychain']=0
settings['threshold']=1e-5
settings['keep_old']=0
settings['gcontrol'] = 0
settings['typegraph'] = 'simple'



if settings['fromggprnd']:
    method = 'GGP'
else:
    method = 'BFRY'

Z = {}
tsigma_range = np.array([0.1,0.3,0.5,0.7])
for i in range(len(tsigma_range)):
    [Z[i], _, _, _, _, _, _, _, _, _]= ggp_dyngraphrnd_sparse(method, talpha, tsigma_range[i], ttau, T, tphi, trho, tgvar, settings)



color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
for t in range(T):
    plt.figure()
    for i in range(len(tsigma_range)):
        plot_degree_sparse(Z[i][t],color=color[i])
    plt.legend((r'$\sigma=0.1$',r'$\sigma=0.3$', r'$\sigma=0.5$', r'$\sigma=0.7$'))
    plt.xlabel('Degree')
    plt.ylabel('Distribution')
    plt.tight_layout()

    plt.savefig('{}/degree_T_{}.png'.format(folder_name,t))


#######################################
# Phi
#######################################

# Folder to save results to

folder_name = 'parameters/phi'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Set the seed
random.seed(1001)

talpha = 1
ttau = 1 # Parameters gamma process
tsigma =0.01
T = 100                       # Number of time steps
trho = 0.1                     # death rate for latent interactions
tgvar = np.ones(T,dtype='int')

settings = {}

settings['dt']=1


settings['L']=20000
settings['fromggprnd']=1
settings['onlychain']=0
settings['threshold']=1e-5
settings['keep_old']=0
settings['gcontrol'] = 0
settings['typegraph'] = 'simple'



if settings['fromggprnd']:
    method = 'GGP'
else:
    method = 'BFRY'
w = {}
indchain = {}
tphi_range = np.array([20,2000])
for i in range(len(tphi_range)):
    [_, w[i], _, _, _, _, _, _, indchain[i], _]= ggp_dyngraphrnd_sparse(method, talpha, tsigma, ttau, T, tphi_range[i], trho, tgvar, settings)

for i in range(len(tphi_range)):
    plt.figure()
    plt.stackplot(np.array(range(T)), w[i][:, indchain[i]].T)
    plt.xlabel('Time')
    plt.ylabel('Weights')
    plt.tight_layout()

    plt.savefig('{}/weights_phi_{}.png'.format(folder_name,tphi_range[i]))

