import numpy as np
import scipy
import warnings

# Gradient
def grad_U(t, mm, wvec, counts, uvec, alpha, sigma, tau, L_BFRY, phi, gv,method):
    #     N = size(M,2)

    # if N ==1
    #     out( :, 1)= -(M(:, 1)) +wvec.*(2*sum(wvec)+2*w_rem + tau)
    # else
    if method == 'Poisson':
        if t == 0:
            t_BFRY = np.float((L_BFRY * sigma / alpha) ** (1 / sigma))
            out= (-(mm + counts[:, t] + uvec - sigma) +np.multiply(wvec,2*gv*np.sum(wvec) + tau + phi + t_BFRY))
        else:
            t_BFRY = np.float((L_BFRY * sigma / alpha) ** (1 / sigma))
            temp =tau+2*gv*np.sum(wvec) +2*phi + t_BFRY
            out = (- (mm + counts[:, t-1] + counts[:, t] + uvec -sigma )+ np.multiply(wvec,temp))

    elif method == 'Exponential':
        if t == 0:
            out= (-(mm + counts[:, t] + 1 - sigma) +np.multiply(wvec,2*gv*np.sum(wvec) + tau + phi + uvec))
        else:
            temp =tau+2*gv*np.sum(wvec) +2*phi + uvec
            out = (- (mm + counts[:, t-1] + counts[:, t] + 1 -sigma )+ np.multiply(wvec,temp))

    else:
        raise ValueError('Method: "{}" Not Implmented'.format(method))

    return out


# Update of the weights
def bfry_sample_weights(iweights, iC, iU, iM, epsilon, alpha, sigma, tau, L_BFRY, phi, gvec, settings):

    # Augmentation method
    method = settings['augment']

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
        u = iU[:, t]
        logw= np.log(w)
        sum_w = np.sum(w)
        sum_uw = np.sum(u*w)
        eps = epsilon[t]

        m = iM[:, t]
        logwprop = logw
        p = np.random.normal(size=(len(w)))

        grad1 = grad_U(t, m, w, counts, u, alpha, sigma, tau, L_BFRY, phi, g, method)
        pprop = p - eps* grad1/2

        for lp in range(L):
            #             wprop = exp(log(wprop) + epsilon*pprop)#wprop = exp(log(wprop) + epsilon*pprop./m_leapfrog)
            logwprop = logwprop + eps*pprop
            if lp<L-1:
                pprop = pprop  - eps* grad_U(t, m, np.exp(logwprop), counts, u, alpha, sigma,  tau, L_BFRY, phi, g, method)

        wprop = np.exp(logwprop)
        pprop = pprop - (eps/2)*grad_U(t, m, wprop, counts, u, alpha, sigma, tau, L_BFRY, phi, g, method)

        sum_wprop = sum(wprop)
        sum_uwprop = np.sum(u * wprop)

        if method == 'Poisson':
            if t == 0:
                # NB np.sum(logwprop) -np.sum(logw) added back in later
                t_BFRY = np.float((L_BFRY * sigma / alpha) ** (1 / sigma))
                temp1 = (- g * (sum_wprop ** 2) + g * (sum_w ** 2)
                         + np.sum(np.multiply(m + u + counts[:, t] - sigma -1, logwprop - logw))
                         - (tau + phi + t_BFRY) * (sum_wprop - sum_w))
            else:
                # NB np.sum(logwprop) -np.sum(logw) added back in later
                t_BFRY = np.float((L_BFRY * sigma / alpha) ** (1 / sigma))
                temp1 = (- g * (sum_wprop ** 2) + g * (sum_w ** 2)
                         + np.sum(np.multiply(m + u + counts[:, t] + counts[:, t - 1] - sigma -1, logwprop - logw))
                         - (tau + 2 * phi + t_BFRY) * (sum_wprop - sum_w))
        elif method == 'Exponential':
            if t==0:
                # NB np.sum(logwprop) -np.sum(logw) added back in later
                temp1 = (- g*(sum_wprop**2) + g*(sum_w**2)
                         + np.sum(np.multiply(m + 1 + counts[:, t]-sigma -1,logwprop - logw) )
                         - (tau+ phi)*(sum_wprop - sum_w) - (sum_uwprop - sum_uw))
            else:
                # NB np.sum(logwprop) -np.sum(logw) added back in later
                temp1 = (- g*(sum_wprop**2) + g*(sum_w**2)
                         + np.sum(np.multiply(m + 1+ counts[:, t] +counts[:, t-1] - sigma -1, logwprop - logw) )
                         - (tau+ 2*phi )*(sum_wprop - sum_w) - (sum_uwprop - sum_uw))

        else:
            raise ValueError('Method: "{}" Not Implmented'.format(method))

        logaccept = temp1 -.5*np.sum(pprop**2-p**2) -np.sum(logw) + np.sum(logwprop)

        if np.isnan(logaccept):
            logaccept = -np.inf
            warnings.warn('logaccept is nan')

        if np.log(np.random.uniform())<logaccept:
            w = wprop

        rate[t] = np.minimum(1, np.exp(logaccept))
        weights_out[:, t]=w

    return [weights_out, rate]



