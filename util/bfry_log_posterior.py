import numpy as np
import scipy
import warnings


# Update of the hyperparameters
def bfry_log_posterior(w, C, u, m, nnew, nold, alpha, sigma, tau, phi, rho, gvec, settings):
    # Calculates the log posterior (up to a normalizing constant)
    # Given the latent counts

    [K, T] = w.shape

    dt = 1

    if 'keep_old' in settings:
        keep_old = settings['keep_old']
    else:
        keep_old = 1

    # TODO: Add in gvec version
    if (gvec != np.ones(T,dtype='int')).all():
        warnings.warn('gvec not yet implemented')



    # First compute the terms involving the latent counts

    lp1_1 = np.sum(m*np.log(w)) - np.sum((np.sum(w,0))**2 - scipy.special.loggamma(0.5*np.sum(m,0) + 1))

    # Add terms if we are keeping the old counts from time t-1 to t

    lp1_2 = 0

    if keep_old:
        # Need to extract nonzero elements of nnew and nold for each t
        # Convert sparse matrices to csr form to do this
        nnew_nz = [None] * T
        nold_nz = [None] * T

        nall = [None] * T
        n_all_minus_old = [None] * T

        nall_nz = [None] * T
        n_all_minus_old_nz = [None] * T

        for t in range(T):
            nnew_nz[t] = nnew[t].tocsr()[nnew[t].tocsr().nonzero()]
            nold_nz[t] = nold[t].tocsr()[nold[t].tocsr().nonzero()]

            nall[t] = nnew[t] + nold[t]
            if t<T-1:
                n_all_minus_old[t] = nall[t] - nold[t+1]
            else:
                n_all_minus_old[t] = nall[t]

            nall_nz[t] = nall[t].tocsr()[nall[t].tocsr().nonzero()]
            n_all_minus_old_nz[t] = n_all_minus_old[t].tocsr()[n_all_minus_old[t].tocsr().nonzero()]


        # Set nold_nz to zero for t=0 (when nold is all-zero sparse)
        nold_nz[0] = [0]

        for t in range(1,T):
            lp1_2 += (np.sum(scipy.special.loggamma(nall_nz[t-1] + 1))
                      - np.sum(scipy.special.loggamma(nold_nz[t] + 1))
                      - np.sum(scipy.special.loggamma(n_all_minus_old_nz[t-1] + 1))
                      - np.sum(nold_nz[t]*rho*dt)
                      + np.sum((n_all_minus_old_nz[t-1])*np.log(1 - np.exp(-rho*dt))))

    log_posterior = lp1_1 + lp1_2

    # Compute all other terms apart from hyperparameter priors

    t_BFRY = np.float((K * sigma / alpha) ** (1 / sigma))

    # Add in the terms for t=1

    lp2_1 = K*(np.log(sigma) - scipy.special.loggamma(1-sigma))

    lp2_2 = np.sum((C[:,0] - sigma)*np.log(w[:,0])) - np.sum((u[:,0] + tau + phi)*w[:,0])

    lp2_3 = np.log(phi)*np.sum(C[:,0]) - np.sum(scipy.special.loggamma(C[:,0] + 1))

    lp2_4 = -K*np.log(((tau + t_BFRY)**sigma-tau**sigma))


    log_posterior += lp2_1 + lp2_2 + lp2_3 + lp2_4

    # Add terms for t>1
    if T>1:
        lp3_1 = np.sum((C[:,1:T] + C[:,:-1] - sigma)*np.log(w[:,1:T])) - np.sum((u[:,1:T] + tau + 2*phi)*w[:,1:T])

        lp3_2 = np.log(phi)*np.sum(C[:,1:T]) - np.sum(scipy.special.loggamma(C[:,1:T] + 1))

        lp3_3 = -np.sum(scipy.special.loggamma(1 - sigma + C[:,:-1]))

        # Calculation of lp3_4 is unstable when C is large
        # Naive calculation works when C==0, treat C>0 case separately

        C_pt = C[:, :-1]
        C_zero_num = np.sum(C_pt == 0)
        C_pt_pos = C_pt[C_pt > 0]

        lp3_4_zero = C_zero_num * (np.log(sigma) - np.log(((tau + phi + t_BFRY) ** (sigma) - (tau + phi) ** (sigma))))

        lp3_4_pos = np.sum(np.log((C_pt_pos - sigma)) - (sigma - C_pt_pos) * np.log(tau + phi)
                           - np.log(1 - ((tau + phi + t_BFRY) / (tau + phi)) ** (sigma - C_pt_pos)))

        lp3_4 = lp3_4_zero + lp3_4_pos

        log_posterior += lp3_1 + lp3_2 + lp3_3 + lp3_4


    # Add in hyperparameter prior terms
    # Constant terms removed, to avoid problems with improper priors

    lp_alpha = (settings['hyper_alpha'][0] - 1) * np.log(alpha) - settings['hyper_alpha'][1] * alpha

    lp_tau = (settings['hyper_tau'][0] - 1) * np.log(tau) - settings['hyper_tau'][1] * tau

    lp_phi = (settings['hyper_phi'][0] - 1) * np.log(phi) - settings['hyper_phi'][1] * phi

    lp_rho = (settings['hyper_rho'][0] - 1) * np.log(rho) - settings['hyper_rho'][1] * rho

    lp_sigma = (settings['hyper_sigma'][0] - 1)* np.log(sigma) + (settings['hyper_sigma'][1] - 1)* np.log(1 - sigma)

    log_posterior += lp_alpha + lp_tau + lp_phi + lp_rho + lp_sigma


    if np.isnan(log_posterior):
        log_posterior = -np.inf
        warnings.warn('log_posterior is nan')

    return log_posterior

