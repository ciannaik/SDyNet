import numpy as np
import scipy


def ggp_sample_C(C, w, phi, alpha, sigma, tau, L):

    # Function that updats the C_{tk} for k=1, ..., K and t=1, ..., N.
    # The last row of C, i.e. C(end,:) is $c_{t\ast}$ for t=1, ..., T.
    # Last column will be for last time point N. No evidence for that. So just
    # sample from prior.

    [K, T] = w.shape
    nb_MH = 1 # Nb of MH iterations

    C_out = C.copy()
    if T>1:
        c = C[:,:-1] # update times 1,....t-1 first

        for nn in range(nb_MH):
            Cnew = np.random.poisson(phi*(w[:, :-1])) # proposal from Poisson distribution

            u = np.random.uniform(size=(K, T-1))
            # acceptance ratio is the ratio of BFRY pdfs, with the proposal and current values of c
            # When c or Cnew is large, acceptance ratio is numerically unstable
            # Naive calculation works when c==0, treat c>0 case separately, same for Cnew

            t_BFRY = np.float((L * sigma / alpha) ** (1 / sigma))

            # Old version
            # logaccept = (np.log((sigma-Cnew)/((tau+phi+t_BFRY)**(sigma-Cnew)-(tau+phi)**(sigma-Cnew)))
            #                  - np.log((sigma-c)/((tau+phi+t_BFRY)**(sigma-c)-(tau+phi)**(sigma-c)))
            #                  + scipy.special.loggamma(1-sigma+c) - scipy.special.loggamma(1-sigma+Cnew)
            #                  + (Cnew - c)*np.log(w[:, 1:]))

            # New version
            Cnew_pos = Cnew[Cnew>0]
            c_pos = c[c>0]

            logaccept = (Cnew - c) * np.log(w[:, 1:])

            logaccept[Cnew==0] += (np.log((sigma)/((tau+phi+t_BFRY)**(sigma)-(tau+phi)**(sigma)))
                                   - scipy.special.loggamma(1-sigma))

            logaccept[Cnew>0] += (np.log((sigma-Cnew_pos)/(((tau+phi+t_BFRY)/(tau+phi))**(sigma-Cnew_pos)- 1))
                                  - (sigma-Cnew_pos)*np.log(tau + phi) - scipy.special.loggamma(1-sigma+Cnew_pos))

            logaccept[c==0] -= (np.log((sigma)/((tau+phi+t_BFRY)**(sigma)-(tau+phi)**(sigma)))
                                - scipy.special.loggamma(1-sigma))

            logaccept[c>0] -= (np.log((sigma-c_pos)/(((tau+phi+t_BFRY)/(tau+phi))**(sigma-c_pos)- 1))
                               - (sigma-c_pos)*np.log(tau + phi) - scipy.special.loggamma(1-sigma+c_pos))



            accept = np.minimum(np.exp(logaccept),1)

            c[u < accept] = Cnew[u < accept]

        C_out[:, :-1] = c
    C_out[:, -1] = np.random.poisson(phi*(w[:, -1]))
    if T>1:
        return [C_out,np.mean(accept)]
    else:
        return [C_out,0]