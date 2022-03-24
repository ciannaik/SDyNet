import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.sparse
from datetime import datetime
from scipy.sparse import csr_matrix, triu
import csv
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '../plotting'))
from plot_degree import plot_degree


# Small dataset (~5,000 nodes)

tsv_file = open("soc-redditHyperlinks-title.tsv")
read_tsv = list(csv.reader(tsv_file, delimiter="\t"))

source = [r[0] for r in read_tsv[1:]]
target = [r[1] for r in read_tsv[1:]]
post_id = [r[2] for r in read_tsv[1:]]
timestamp = [r[3] for r in read_tsv[1:]]
weight = np.array([int(r[4]) for r in read_tsv[1:]])

read_tsv = 0



unique_nodes = list(set(source+target))



source_id = np.zeros(len(source),dtype=int)
target_id = np.zeros(len(target),dtype=int)

unique_dict = {}
for i,u in enumerate(unique_nodes):
    unique_dict[u] = i

for i,s in enumerate(source):
    source_id[i] = unique_dict[s]

for i,t in enumerate(target):
    target_id[i] = unique_dict[t]

timestamp_year = np.array([datetime.strptime(t,'%Y-%m-%d %H:%M:%S').year for t in timestamp])
timestamp_month = np.array([datetime.strptime(t,'%Y-%m-%d %H:%M:%S').month for t in timestamp])
timestamp_day = np.array([datetime.strptime(t,'%Y-%m-%d %H:%M:%S').day for t in timestamp])



ones = np.ones(len(source_id), np.uint32)
n = np.maximum(np.max(source_id),np.max(target_id))+1


G_all = scipy.sparse.coo_matrix((ones, (source_id, target_id)),shape=(n,n))
G_all = G_all+scipy.sparse.coo_matrix.transpose(G_all)
deg = scipy.sparse.csr_matrix.sum(G_all.tocsr(),0)
any = np.squeeze(np.asarray(deg>0))
G_all = G_all[any,:]
G_all = G_all[:, any]
G_all.shape


plt.figure()
plt.hist(timestamp_year,1000)
plt.show()

data=np.array((source_id,target_id)).T

# Remove initial set of nodes
ind = timestamp_year ==2016
times = timestamp_month[ind] - 1
data = data[ind,:]
source = np.asarray(source)[ind].tolist()
target = np.asarray(target)[ind].tolist()

source_id = source_id[ind]
target_id = target_id[ind]

ind = np.argsort(times)
data = data[ind,:]
source = np.asarray(source)[ind].tolist()
target = np.asarray(target)[ind].tolist()

source_id = source_id[ind]
target_id = target_id[ind]

ones = np.ones(data.shape[0], np.uint32)
n = np.maximum(np.max(data[:,:2]),np.max(data[:,:2]))+1


G_all = scipy.sparse.coo_matrix((ones, (data[:,0], data[:,1])),shape=(n,n))
G_all = G_all+scipy.sparse.coo_matrix.transpose(G_all)
deg = scipy.sparse.csr_matrix.sum(G_all.tocsr(),0)
any = np.squeeze(np.asarray(deg>0))
G_all = G_all[any,:]
G_all = G_all[:, any]
G_all.shape
# deg = scipy.sparse.csr_matrix.sum(G_all.tocsr(),0)

unique_nodes = list(set(source+target))
unique_nodes = np.asarray(unique_nodes)
# unique_nodes[np.asarray(np.argsort(deg))[0]][-5:]

#
# f = open('correct_2016_words.pckl', 'wb')
# pickle.dump([unique_nodes], f)
# f.close()

h = plot_degree(csr_matrix.sign(G_all))

# Set number of time steps
T = 12
# Array to hold final adjacency matrices
G = {}
tG = {}
Nnew = {}

for i in range(T):
    print(i)
    ind = np.nonzero(times == i)
    ones = np.ones(data[ind, 1].shape[1], np.uint32)
    Gtemp = scipy.sparse.coo_matrix((ones, (np.squeeze(data[ind,0]), np.squeeze(data[ind,1]))), shape=(n, n))
    Gtemp = Gtemp + scipy.sparse.coo_matrix.transpose(Gtemp)
    # deg = scipy.sparse.csr_matrix.sum(Gtemp.tocsr(), 0)
    # any = np.squeeze(np.asarray(deg > 0))
    Gtemp = Gtemp[any, :]
    Gtemp = Gtemp[:, any]
    Gtemp = scipy.sparse.coo_matrix.sign(Gtemp)
    G[i] = Gtemp  # symmetric

    Ntemp = scipy.sparse.coo_matrix((ones, (np.squeeze(data[ind,0]), np.squeeze(data[ind,1]))), shape=(n, n))
    Ntemp = Ntemp + scipy.sparse.coo_matrix.transpose(Ntemp)
    # deg = scipy.sparse.csr_matrix.sum(Ntemp.tocsr(), 0)
    # any = np.squeeze(np.asarray(deg > 0))
    Ntemp = Ntemp[any, :]
    Ntemp = Ntemp[:, any]
    Nnew[i] = Ntemp

    print(np.sum(G[i]/2))

# for i in range(len(timestamps)-1):
#     G[i] = G[i] + scipy.sparse.csr_matrix.transpose(G[i])
#     G[i] = G[i][any, :]
#     G[i] = G[i][:, any]

for i in range(T):
     tG[i]=scipy.sparse.csr_matrix(scipy.sparse.triu(G[i], 1))



f = open('store_reddit.pckl', 'wb')
pickle.dump([tG, G_all, np.unique(times), G, Nnew, unique_nodes], f)
f.close()
