import numpy as np
import scipy
from scipy.sparse import csr_matrix
from tpoissonrnd import tpoissrnd
from k_tpoissonrnd import k_tpoissrnd
from tbinomrnd import tbinomrnd

import warnings


def bfry_sample_interaction_counts(t, Zt, logw, Nnew, Nold, ind1, ind2, N, gv, rho, settings):
    # Metropolis- Hastings. Jointly update mnew and mold
    # all output matrices are in upper triangular form.

    k = logw.shape[0]
    mnew_t = Nnew[t]
    mold_t = Nold[t]

    deltat = settings['dt']
    if t<N-1:

        if t == 0:
            mnew_mt = csr_matrix((k, k),dtype='int')
            mold_mt = csr_matrix((k, k),dtype='int')
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

        mnew = mnew_t[id]
        mold = mold_t[id]
        mold_plt = mold_plust[id]

        # keyboard
        lograte_poi = np.log(2) + np.log(gv)+logw[ind1] + logw[ind2]
        lograte_poi[ind1==ind2] = np.log(gv)+2*logw[ind1[ind1==ind2]]

        threshold = 0# mold_plust(id)-mold_prop

        ############################
        nd_new= threshold +np.random.poisson(np.exp(lograte_poi))
        td_new = threshold + tpoissrnd(np.exp(lograte_poi))
        #count = sparse(ind1, ind2, d, k, k)

        mnew_prop = np.multiply(td_new,(mpt==0)) + np.multiply(nd_new,(mpt>0))
        ############################
        # mnew_prop = k_tpoissrnd(np.squeeze(np.asarray(np.exp(lograte_poi))),np.squeeze(np.asarray(mpt - mold))).reshape(1,len(id[0]))


        nd_old= np.random.binomial(mpt, pi*np.ones(len(id[0])))
        td_old = tbinomrnd(np.squeeze(np.asarray(mpt)), pi*np.ones(len(id[0]))).reshape(1,len(id[0]))

        mold_prop = np.multiply(td_old,(mnew_prop==0)) + np.multiply(nd_old,(mnew_prop>0))


        proposal = mnew_prop + mold_prop
        current = mnew + mold

        if pi==0:
            pi =0.9999

        aa=current - mold_plt +1
        bb=proposal-mold_plt+1


        # Separate the case where pi  = 1, i.e. rho = 0
        # Here we sample from the true distribution
        logaccept = (scipy.special.loggamma(proposal+1) +scipy.special.loggamma(aa)
                     - scipy.special.loggamma(current+1)-scipy.special.loggamma(bb)
                     + (proposal - current)*np.log(1-pi))

        u = np.random.uniform(size=len(id[0]))
    #     t
        accept = np.squeeze(np.asarray(np.exp(logaccept)))
        accept[np.isnan(accept)] = 0

        mnew = np.squeeze(np.asarray(mnew))
        mold = np.squeeze(np.asarray(mold))

        mnew_prop = np.squeeze(np.asarray(mnew_prop))
        mold_prop = np.squeeze(np.asarray(mold_prop))

        mnew[u<accept] = mnew_prop[u<accept]
        mold[u<accept] = mold_prop[u<accept]

        acceptance_rate = np.mean(u<accept)

    else:
        if N==1:
            mnew_mt = csr_matrix((k, k),dtype='int')
            mold_mt = csr_matrix((k, k),dtype='int')

        else:
            mnew_mt = Nnew[t-1]

            mold_mt = Nold[t-1]

        pi = np.exp(-rho*deltat)

        id = np.nonzero(Zt)  # Zt should be in the upper triangular form

        mpt = mnew_mt[id] + mold_mt[id]

        # keyboard
        lograte_poi = np.log(2) + np.log(gv) + logw[ind1] + logw[ind2]
        lograte_poi[ind1 == ind2] = np.log(gv) + 2 * logw[ind1[ind1 == ind2]]

        threshold = 0  # mold_plust(id)-mold_prop
        nd_new = threshold + np.random.poisson(np.exp(lograte_poi))
        td_new = threshold + tpoissrnd(np.exp(lograte_poi))
        # count = sparse(ind1, ind2, d, k, k)

        mnew = np.multiply(td_new, (mpt == 0)) + np.multiply(nd_new, (mpt > 0))
        # mnew = k_tpoissrnd(np.squeeze(np.asarray(np.exp(lograte_poi))),np.squeeze(np.asarray(mpt - mold_mt[id]))).reshape(1,len(id[0]))


        nd_old = np.random.binomial(mpt, pi * np.ones(len(id[0])))
        td_old = tbinomrnd(np.squeeze(np.asarray(mpt)), pi*np.ones(len(id[0]))).reshape(1,len(id[0]))

        mold = np.multiply(td_old, (mnew == 0)) + np.multiply(nd_old, (mnew > 0))

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

    return [mnew_res, mold_res, mn,acceptance_rate]