import numpy as np
from matplotlib import pyplot as plt

def plot_degree(G, step=1, color='b'):

    # Calculate Degrees
    deg = np.squeeze(np.sum(G,0))
    # Remove nodes with no connections
    any = np.squeeze(np.asarray(deg > 0))
    G = G[any, :]
    G = G[:, any]
    deg = np.squeeze(np.sum(G,0))

    # Uses logarithmic binning to get a less noisy estimate of the
    # pdf of the degree distribution

    edgebins = 2**np.arange(0,17,step)
    sizebins = edgebins[1:] - edgebins[:-1]

    sizebins = np.append(sizebins, 1)
    centerbins = edgebins
    counts = np.histogram(deg, np.append(edgebins,np.inf))
    freq = np.divide(counts[0],sizebins)/G.shape[0]
    h2 = plt.loglog(centerbins, freq,'o',color=color)
    plt.xlabel('Degree', fontsize=16)
    plt.ylabel('Distribution', fontsize=16)
    plt.gca().set_xlim(left=1)


    return [h2, centerbins, freq]