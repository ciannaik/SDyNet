import numpy as np
import scipy
import warnings
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
    csr_matrix, isspmatrix, dok_matrix, lil_matrix, bsr_matrix)

def ggp_sample_gvar_mh(weights, nnew, gvar, g_a, g_b):


    [K, T] = weights.shape
    lam = 0.1

    for t in range(T):
        gcur = gvar[t]
        gprop = np.multiply(gvar[t],np.exp(lam*np.random.normal()))
        w = np.squeeze(weights[:,t])
        nnew_t = nnew[t]
        # wmtx = csr_matrix(np.multiply(np.outer(w,w),(np.triu(np.ones((K,K)),1))))

        # Old version - needs checking
        # logaccept =np.sum( np.multiply(nnew_t,(np.log(gprop) - np.log(gcur)))- 2*np.multiply(wmtx,(gprop - gcur)))

        # New version
        logaccept =0.5*np.sum( np.multiply(nnew_t,(np.log(gprop) - np.log(gcur)))) - np.multiply(0.5*(np.sum(w)**2-np.sum(w**2)),(gprop - gcur))




        logaccept = logaccept -g_b*(gprop - gcur) + g_a*(np.log(gprop) - np.log(gcur))

        if np.log(np.random.uniform())<logaccept:
            gvar[t] = gprop

    return gvar