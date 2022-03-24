import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt
import scipy.sparse
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../plotting'))
from plot_degree_post_pred import *

filename = 'Days.net'

data = np.zeros((243447,4),dtype=int)
with open(filename) as f:
    count = 0
    for line in itertools.islice(f, 13334, None):
       (node1, node2, weight, time) = line.split()
       data[count,0] = np.int(node1)
       data[count,1] = np.int(node2)
       data[count,2] = np.int(weight)
       data[count,3] = eval(time)[0]

       count += 1
words = {}
with open(filename) as f:
    for line in itertools.islice(f, 1, 13333):
       (key, val,_) = line.split()
       words[int(key)] = val.strip('"')


n = np.max(np.max(data[:,:2]))
data[:,[0,1,3]] = data[:,[0,1,3]] - 1
time = data[:,3]

plt.figure()
plt.hist(time)
plt.show()

########################
# Convert from days to weeks

time = np.floor(time/7).astype(int)
data[:,3] = np.floor(data[:,3]/7).astype(int)

ind = time<=8

data = data[ind, :]
time = time[ind]


########################



ind = np.argsort(time)

data = data[ind, :]
time = time[ind]
ones = np.ones(len(data[:,1]), np.uint32)
weights = data[:,2]


G_all = scipy.sparse.coo_matrix((ones, (data[:,0], data[:,1])),shape=(n,n))
G_all = G_all+scipy.sparse.coo_matrix.transpose(G_all)
deg = scipy.sparse.csr_matrix.sum(G_all.tocsr(),0)

G_all.shape
h = plot_degree(G_all)


# Set number of time steps
T = np.max(time)
# Array to hold final adjacency matrices
G = {}
tG = {}
Nnew = {}

for i in range(T):
    print(i)
    ind = np.nonzero(time == i)
    ones = np.ones(data[ind, 1].shape[1], np.uint32)
    Gtemp = scipy.sparse.coo_matrix((ones, (np.squeeze(data[ind,0]), np.squeeze(data[ind,1]))), shape=(n, n))

    Gtemp = Gtemp + scipy.sparse.coo_matrix.transpose(Gtemp)
    # Gtemp = Gtemp[any, :]
    # Gtemp = Gtemp[:, any]
    Gtemp = scipy.sparse.coo_matrix.sign(Gtemp)
    G[i] = Gtemp  # symmetric

    Ntemp = scipy.sparse.coo_matrix((np.squeeze(data[ind,2]), (np.squeeze(data[ind,0]), np.squeeze(data[ind,1]))), shape=(n, n))
    Ntemp = Ntemp + scipy.sparse.coo_matrix.transpose(Ntemp)

    Nnew[i] = Ntemp

    print(np.sum(G[i]/2))

for i in range(T):
     tG[i]=scipy.sparse.csr_matrix(scipy.sparse.triu(G[i], 1))



f = open('store_reuters.pckl', 'wb')
pickle.dump([tG, G_all, np.unique(time), G, Nnew, words], f)
f.close()
