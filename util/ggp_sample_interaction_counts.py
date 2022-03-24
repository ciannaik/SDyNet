import numpy as np
import scipy
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
    csr_matrix, isspmatrix, dok_matrix, lil_matrix, bsr_matrix)
from tpoissonrnd import tpoissrnd
import warnings


def ggp_sample_interaction_counts(t, Zt, logw, Nnew, Nold, ind1, ind2, N, gv, rho, settings):
    # Metropolis- Hastings. Jointly update mnew and mold
    # all output matrices are in upper triangular form.

    k = logw.shape[0]
    mnew_t = Nnew[t]
    mold_t = Nold[t]

    deltat = settings['dt']

    if t<N-1:

        if t == 0:
            mnew_mt = csr_matrix(np.zeros((k, k),dtype='int'))
            mold_mt = csr_matrix(np.zeros((k, k),dtype='int'))
        else:
            mnew_mt = Nnew[t-1]
            mold_mt = Nold[t-1]

        mold_plust = Nold[t+1]



        pi = np.exp(-rho*deltat)
        if pi==0:
            pi =0.9999

        id = np.nonzero(Zt)# Zt should be in the upper triangular form

        # keyboard
        mpt = mnew_mt[id] + mold_mt[id]
        mold_prop = np.random.binomial(mpt, pi*np.ones((1,len(id[0]))))

        # keyboard
        lograte_poi = np.log(2) + np.log(gv)+logw[ind1] + logw[ind2]
        lograte_poi[ind1==ind2] = np.log(gv)+2*logw[ind1[ind1==ind2]]

        threshold = 0# mold_plust(id)-mold_prop
        nd = threshold +np.random.poisson(np.exp(lograte_poi))
        td = threshold + tpoissrnd(np.exp(lograte_poi))

        #count = sparse(ind1, ind2, d, k, k)

        mnew = mnew_t[id]
        mold = mold_t[id]
        mold_plt = mold_plust[id]
        mnew_prop = np.multiply(td,(mold_prop==0)) + np.multiply(nd,(mold_prop>0))


        proposal = mnew_prop + mold_prop
        current = mnew + mold

        if pi==0:
            pi =0.9999

        aa=current - mold_plt +1
        bb=proposal-mold_plt+1

        logaccept = (scipy.special.loggamma(proposal+1) +scipy.special.loggamma(aa)
                     - scipy.special.loggamma(current+1)-scipy.special.loggamma(bb)
                     + (proposal - current)*np.log(1-pi))

        u = np.random.uniform(size=len(id[0]))
    #     t
        accept = np.squeeze(np.asarray(np.exp(logaccept)))
        accept[np.isnan(accept)]=0
    #     keyboard

        mnew = np.squeeze(np.asarray(mnew))
        mold = np.squeeze(np.asarray(mold))

        mnew_prop = np.squeeze(np.asarray(mnew_prop))
        mold_prop = np.squeeze(np.asarray(mold_prop))

        mnew[u<accept] = mnew_prop[u<accept]
        mold[u<accept] = mold_prop[u<accept]

        acceptance_rate = np.mean(u < accept)

    else:
        if N==1:
            mnew_mt = csr_matrix(np.zeros((k, k),dtype='int'))
            mold_mt = csr_matrix(np.zeros((k, k),dtype='int'))

        else:
            mnew_mt = Nnew[t-1]

            mold_mt = Nold[t-1]

        pi = np.exp(-rho*deltat)

        id = np.nonzero(Zt)  # Zt should be in the upper triangular form

        mpt = mnew_mt[id] + mold_mt[id]
        mold = np.random.binomial(mpt, pi * np.ones(len(id[0])))

        lograte_poi = np.log(2) + np.log(gv) + logw[ind1] + logw[ind2]
        lograte_poi[ind1 == ind2] = np.log(gv) + 2 * logw[ind1[ind1 == ind2]]
        threshold = 0  # mold_plust(id)-mold_prop
        nd = threshold + np.random.poisson(np.exp(lograte_poi))
        td = threshold + tpoissrnd(np.exp(lograte_poi))
        mnew = np.multiply(td,(mold==0)) + np.multiply(nd,(mold>0))

        mnew = np.squeeze(np.asarray(mnew[0, :]))
        mold = np.squeeze(np.asarray(mold[0, :]))

        acceptance_rate = 1

    #    mnew_res= sparse(k, k)
    #   mold_res= sparse(k, k)
    mnew_res = csr_matrix((mnew, (ind1, ind2)), shape=(k,k))
    mold_res = csr_matrix((mold, (ind1, ind2)), shape=(k,k))

    #      mnew_res(sub2ind(size(mnew_res), ind1', ind2')) = mnew
    #     mold_res(sub2ind(size(mnew_res), ind1', ind2')) = mold

    mn = (np.transpose(np.squeeze(np.asarray(np.sum(mnew_res,0)))) + np.squeeze(np.asarray(np.sum(mnew_res,1))))

    if sum(mn<0):
        warnings.warn('mn<0')

    return [mnew_res, mold_res, mn, acceptance_rate]