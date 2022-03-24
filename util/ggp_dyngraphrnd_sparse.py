from GGPrnd import GGPrnd
import numpy as np
from exptiltBFRY import exptiltBFRY
from exptiltBFRY_reparam import exptiltBFRY_reparam
from scipy.sparse import csr_matrix, lil_matrix, find


def ggp_dyngraphrnd_sparse(typeweights, alpha, sigma, tau, T, phi, rho, gvar, settings):

    dt = settings['dt']
    # Function that simulates from the process.
    # output Z: symmetric
    #        N's: all directed counts, non-symmetric

    threshold =  settings['threshold']

    if 'keep_old' in settings:
        keep_old = settings['keep_old']
    else:
        keep_old = 1

    binall = []

    if typeweights == "GGP":  # power law with exp sigma

        indmax = 2000000 # so that to allocate memory
        c = np.zeros((T, indmax),dtype='int')
        w = np.zeros((T, indmax),dtype='float')
        K = np.zeros(T,dtype='int')
        # Sample the graph conditional on the weights w

        wvec = GGPrnd(alpha, sigma, tau,  threshold)
        w[0, :len(wvec)] = wvec



        c[0, :len(wvec)] = np.random.poisson(phi*wvec)
        K[0] = len(wvec)
        for t in range(1,T):
            print(t)
            # Sample counts for existing atoms
            ind = c[t-1,:]>0
            w[t, ind] = np.random.gamma(np.maximum(c[t-1,ind]-sigma, 0 ), 1/(tau+phi))
            c[t,ind] = np.random.poisson(phi*w[t, ind])

            # Sample new atoms
            wnew = GGPrnd(alpha, sigma, tau+phi, threshold)

            cnew = np.random.poisson(phi*wnew)
            knew = len(wnew)
            w[t, K[t-1]:K[t-1]+ knew]=wnew
            c[t, K[t-1]:K[t-1]+ knew]=cnew
            K[t] = K[t-1]+ knew

        size_w=K[-1]


        w=w[:,:size_w]
        c=c[:,:size_w]

    elif typeweights == "BFRY":  # power law with exp sigma but fixed numb of nodes L
        L = settings['L'] # Set number of nodes generated

        c = np.zeros((T, L), dtype='int')
        w = np.zeros((T, L), dtype='float')
        # Sample the graph conditional on the weights w

        wvec = exptiltBFRY(alpha, sigma, tau, L)
        w[0, :] = wvec

        c[0, :] = np.random.poisson(phi * wvec)
        for t in range(1, T):
            print(t)
            # Sample counts for all atoms
            t_BFRY = np.float((L * sigma / alpha) ** (1 / sigma))
            wvec = exptiltBFRY_reparam(t_BFRY, sigma-c[t-1, :], tau + phi, L)
            w[t, :] = wvec
            c[t, :] = np.random.poisson(phi * w[t, :])

        size_w = L
        K = L

    # # Sample the network given the C's

    N_new = [None]*T
    N_old = [None]*T
    N = [None]*T
    Z = [None]*T
    M = [None]*T


    # t=1
    N_new[0] = lil_matrix((size_w, size_w),dtype='int')
    N_old[0] = lil_matrix((size_w, size_w),dtype='int')
    N[0] = lil_matrix((size_w, size_w),dtype='int')
    Z[0] = lil_matrix((size_w, size_w),dtype='int')
    M = np.zeros((size_w, T), dtype='int')

    cumsum_w = np.append(0, np.cumsum(w[0,:]))
    W_star = cumsum_w[-1] # Total mass of the GGP
    D_star = np.random.poisson(gvar[0]*W_star**2) # Total number of directed edges

    temp = W_star * np.random.uniform(size = (D_star, 2))
    bin = np.digitize(temp, cumsum_w) - 1
    # Want values outside either side of the range of w to give a value of 0
    bin[bin == len(cumsum_w)] = -1
    for d in range(D_star):
        N_new[0][bin[d,0], bin[d,1]] = N_new[0][bin[d,0], bin[d,1]]+ 1

    binall = np.append(binall ,bin.flatten())
    N[0] = N_new[0]
    Z[0] = (N[0]+np.transpose(N[0]))>0


    # indall=[]
    # indall = [indall unique(bin(:)) ]


    for t in range(1,T):
        print(t)

        N_new[t] = lil_matrix((size_w, size_w), dtype='int')
        N_old[t] = lil_matrix((size_w, size_w), dtype='int')
        N[t] = lil_matrix((size_w, size_w), dtype='int')
        Z[t] = lil_matrix((size_w, size_w), dtype='int')

        cumsum_w = np.append(0, np.cumsum(w[t, :]))
        W_star = cumsum_w[-1]  # Total mass of the GGP
        D_star = np.random.poisson(gvar[t] * W_star ** 2)  # Total number of directed edges

        temp = W_star * np.random.uniform(size=(D_star, 2))
        bin = np.digitize(temp, cumsum_w) - 1
        # Want values outside either side of the range of w to give a value of 0
        bin[bin == len(cumsum_w)] = -1
        binall = np.append(binall, bin.flatten())

        for d in range(D_star):
            N_new[t][bin[d, 0], bin[d, 1]] = N_new[t][bin[d, 0], bin[d, 1]] + 1

        # Sample old interactions
        if keep_old:
            N_old[t][find(N[t-1])[:2]] = np.random.binomial(find(N[t-1])[2], np.exp(-rho*dt) )

        # Aggregate old + new
        N[t] = N_new[t] + N_old[t]

        # Obtain undirected graph from directed one
        Z[t] = (N[t] + np.transpose(N[t])) > 0


    for t in range(T):
        M[:, t] = np.ndarray.flatten(np.array(N_new[t].sum(axis=0))) + np.ndarray.flatten(np.array(N_new[t].sum(axis=1))) - N_new[t].diagonal()
    #     keyboard
    indlinks = np.sum(M, 1)>0 # indices of the nodes that participate in links


    [Nall,T]=np.shape(M)
    # indcnz = (np.sum(M,1)==0 & np.transpose(np.sum(c[:len(c)-1,:],0))>0)

    indcnz = np.zeros((Nall, T-1)) # Nall x T-1

    for t in range(T-1):
        indcnz[:, t] = np.transpose(c[t,:])>0 & (np.sum(M,1)==0)

    indcnz = np.append(np.zeros((Nall,1)) ,indcnz) # Nall x T

    # Convert matrices to csr for output

    for t in range(T):
        N_new[t] = N_new[t].tocsr()
        N_old[t] = N_old[t].tocsr()
        N[t] = N[t].tocsr()

    return [Z, w, c, K, N_new, N_old, N, M, indlinks, indcnz]