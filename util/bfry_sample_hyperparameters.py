import numpy as np
import scipy
import warnings


# Update of the hyperparameters
def bfry_sample_hyperparameters(w, C, u, alpha, sigma, tau, phi, settings):
    # Input:
    #       w: the matrix of the KXT weights excluding the w_rem masses.
    #       settings.rw_std vector of length 2. Contains the std hyper for the
    #       sigma and tau correspondingly.
    #

    [K, T] = w.shape

    alpha_out = alpha
    sigma_out = sigma
    tau_out = tau
    phi_out = phi

    nbMH=1
    for nn in range(nbMH):
        sum_w = np.sum(w,0)
        # Sample (alpha,sigma,tau,phi) from the proposal distribution
        if settings['estimate_alpha']:
            alphaprop = np.exp(np.log(alpha) + settings['rw_std'][0]*np.random.normal())
        else:
            alphaprop = alpha

        if settings['estimate_sigma']:
            zz =(np.log(sigma/(1-sigma)) + settings['rw_std'][1]*np.random.normal())
            sigmaprop = 1/(1 + np.exp(-zz))

            if sigmaprop<0 or sigmaprop>1:
                warnings.warn('sigma outside (0,1)')
        else:
            sigmaprop = sigma

        if settings['estimate_tau']:
            tauprop = np.exp(np.log(tau) + settings['rw_std'][2]*np.random.normal())
        else:
            tauprop = tau

        if settings['estimate_phi']:
            phiprop = np.exp(np.log(phi) + settings['rw_std'][3]*np.random.normal())
        else:
            phiprop = phi

        # Compute the acceptance probability
        # First, compute the fixed terms, i.e. the terms of the ratio without
        # that do not correspond to the prior or proposal of \alpha, \sigma
        # \tau and \phi

        # First compute the terms involving only t=1

        t1_1 = K*(np.log(sigmaprop)-np.log(sigma))

        t1_2 = K*(scipy.special.loggamma(1-sigma) - scipy.special.loggamma(1-sigmaprop))

        t_BFRY = np.float((K * sigma / alpha) ** (1 / sigma))
        t_BFRY_prop = np.float((K * sigmaprop / alphaprop) ** (1 / sigmaprop))

        t1_3 = -(tauprop - tau +phiprop-phi +t_BFRY_prop - t_BFRY)*sum_w[0]

        t1_4 = K*np.log(((tau+t_BFRY)**sigma-tau**sigma)/((tauprop+t_BFRY_prop)**sigmaprop-tauprop**sigmaprop))

        logaccept = t1_1 + t1_2 + t1_3 + t1_4

        # Add terms for t=2,...,T

        if T>1:
            t2_1 = np.sum(np.log(scipy.special.gamma(1-sigma + C[:,1:T])/
                                 scipy.special.gamma(1-sigmaprop + C[:,1:T])))

            t2_2 = -(tauprop - tau +2*phiprop-2*phi +t_BFRY_prop - t_BFRY)* np.sum(sum_w[1:T])

            t2_3 = np.sum(np.log((sigmaprop - C[:,1:T])/(sigma - C[:,1:T])))

            t2_4 = np.sum(np.log(((tau+phi+t_BFRY)**(sigma-C[:,1:T])-(tau+phi)**(sigma-C[:,1:T]))/
                                 ((tauprop+phiprop+t_BFRY_prop)**(sigmaprop-C[:,1:T])
                                  -(tauprop+phiprop)**(sigmaprop-C[:,1:T]))))

            logaccept = logaccept + t2_1 + t2_2 + t2_3 + t2_4

        # Add terms for all t=1,...,T

        t3_1 = (sigma - sigmaprop)*np.sum(np.log(w))

        if phi != 0:
            t3_2 = np.sum(C)*(np.log(phiprop)-np.log(phi))
        else:
            t3_2 = np.log(phiprop**np.sum(C)) - np.log(phi**np.sum(C))

        t3_3 = np.sum(u)*(np.log(t_BFRY_prop)-np.log(t_BFRY))

        logaccept = logaccept + t3_1 + t3_2 + t3_3

        if settings['estimate_alpha']:
            logaccept = logaccept + settings['hyper_tau'][0] * (np.log(alphaprop) - np.log(alpha)) \
                        - settings['hyper_tau'][1] * (alphaprop - alpha)

        if settings['estimate_tau']:
            logaccept = logaccept + settings['hyper_tau'][0] * (np.log(tauprop) - np.log(tau)) \
                        - settings['hyper_tau'][1] * (tauprop - tau)

        if settings['estimate_phi']:
            logaccept = logaccept + settings['hyper_tau'][0] * (np.log(phiprop) - np.log(phi)) \
                        - settings['hyper_tau'][1] * (phiprop - phi)

        if settings['estimate_sigma']:
            logaccept = logaccept + settings['hyper_sigma'][0]*(np.log(sigmaprop) - np.log(sigma)) \
                        + settings['hyper_sigma'][1]*(np.log(1-sigmaprop) - np.log(1-sigma))

        if np.isnan(logaccept):
            logaccept = -np.inf
            warnings.warn('logaccept is nan')

        # Accept step

        if np.log(np.random.uniform())<logaccept:

            alpha_out = alphaprop
            sigma_out = sigmaprop
            tau_out = tauprop
            phi_out = phiprop

    if logaccept>1000:
        warnings.warn('logaccept = {}'.format(logaccept))

    rate2 = np.minimum(1, np.exp(logaccept))
    #keyboard
    return [alpha_out, sigma_out, tau_out, phi_out, rate2]

