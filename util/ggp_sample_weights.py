import numpy as np
import scipy
import warnings

# Gradient
def grad_U(t, mm, wvec, counts, alpha, sigma, tau, L_BFRY, phi, gv):
    #     N = size(M,2)

    # if N ==1
    #     out( :, 1)= -(M(:, 1)) +wvec.*(2*sum(wvec)+2*w_rem + tau)
    # else
    if t == 0:
        # out= -(mm + max(counts( :,  t)-sigma, 0)) +wvec.*(2*gv*sum(wvec)+2*gv*w_rem +tau + phi)
        t_BFRY = np.float((L_BFRY * sigma / alpha) ** (1 / sigma))
        out= (-(mm + counts[:, t]-sigma) +np.multiply(wvec,2*gv*np.sum(wvec) + tau + phi)
              + t_BFRY*wvec/(np.exp(t_BFRY*wvec)-1))
    else:
        temp =tau+2*gv*np.sum(wvec) +2*phi
        t_BFRY = np.float((L_BFRY * sigma / alpha) ** (1 / sigma))
        # out = - (mm + max(counts(:, t-1) + counts(:, t)-sigma, 0) )+ wvec.*temp
        out = (- (mm + counts[:, t-1] + counts[:, t]-sigma )+ np.multiply(wvec,temp)
               + t_BFRY*wvec/(np.exp(t_BFRY*wvec)-1))

    return out


# Update of the weights
def ggp_sample_weights(iweights, iC, iM, epsilon, alpha, sigma, tau, L_BFRY, phi, gvec, settings):

    [K, T] = iweights.shape
    rate = np.zeros(T)

    # Leapfrog steps
    L=settings['leapfrog']['L']

    weights_out = iweights.copy()
    # exclude the rem counts from computation. Not needed
    # Update weights
    for t in range(T):
        g=gvec[t]
        counts= iC

        w = iweights[:, t]
        logw= np.log(w)
        sum_w = np.sum(w)
        eps = epsilon[t]

        m = iM[:, t]
        logwprop = logw
        p = np.random.normal(size=(len(w)))

        grad1 = grad_U(t, m, w, counts, alpha, sigma, tau, L_BFRY, phi, g)
        pprop = p - eps* grad1/2

        for lp in range(L):
            #             wprop = exp(log(wprop) + epsilon*pprop)#wprop = exp(log(wprop) + epsilon*pprop./m_leapfrog)
            logwprop = logwprop + eps*pprop
            if lp<L-1:
                pprop = pprop  - eps* grad_U(t, m, np.exp(logwprop), counts, alpha, sigma,  tau, L_BFRY, phi, g)

        wprop = np.exp(logwprop)
        pprop = pprop - (eps/2)*grad_U(t, m, wprop, counts, alpha, sigma, tau, L_BFRY, phi, g)

        sum_wprop = sum(wprop)

        if t==0:
            # NB np.sum(logwprop) -np.sum(logw) added back in later
            t_BFRY = np.float((L_BFRY * sigma / alpha) ** (1 / sigma))
            temp1 = (- g*(sum_wprop**2) + g*(sum_w**2)
                     + np.sum(np.multiply(m + counts[:, t]-sigma -1,logwprop - logw) )
                     - (tau+ phi)*(sum_wprop - sum_w)
                     + np.sum(np.log(1-np.exp(-t_BFRY*wprop))-np.log(1-np.exp(-t_BFRY*wprop))))
        else:
            # NB np.sum(logwprop) -np.sum(logw) added back in later
            t_BFRY = np.float((L_BFRY * sigma / alpha) ** (1 / sigma))
            temp1 = (- g*(sum_wprop**2) + g*(sum_w**2)
                     + np.sum(np.multiply(m + counts[:, t] +counts[:, t-1] - sigma -1, logwprop - logw) )
                     - (tau+ 2*phi)*(sum_wprop - sum_w)
                     + np.sum(np.log(1-np.exp(-t_BFRY*wprop))-np.log(1-np.exp(-t_BFRY*wprop))))

        logaccept = temp1 -.5*np.sum(pprop**2-p**2) -np.sum(logw) + np.sum(logwprop)

        if np.isnan(logaccept):
            logaccept = -np.inf
            warnings.warn('logaccept is nan')

        if np.log(np.random.uniform())<logaccept:
            w = wprop

        rate[t] = np.minimum(1, np.exp(logaccept))
        weights_out[:, t]=w

    return [weights_out, rate]



