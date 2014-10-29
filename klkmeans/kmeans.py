#-*- coding: utf8
from __future__ import division, print_function

import numpy as np

def _compute_centroids(X, assign, num_clusters):
    C = np.zeros(shape=(num_clusters, X.shape[1]), dtype='d')
    for k in xrange(num_clusters):

        if not (assign == k).any():
            continue

        K = X[assign == k]
        if K.ndim == 1:
            K = K[np.newaxis]
        C[k] = X[assign == k].mean(axis=0) 
    return C

def _surprisal_mat(X):
    #Some elements have zero prob, ingore and treat warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        L = np.log2(X)
        L[np.isnan(X)] = 0
        L[np.isinf(X)] = 0
    return L

def _dist_all(X, C):
    S_x = _surprisal_mat(X)
    S_c = _surprisal_mat(C)
    
    D = (X * (S_x - S_c[:, np.newaxis,: ])).sum(axis=2).T
    return D

def _base_kmeans(X, C, n_iters=-1):
    
    num_clusters = C.shape[0]
    n = X.shape[0]

    C_final = C

    #KMeans algorithm
    cent_dists = None
    assign = None
    prev_assign = None
    best_shift = None

    iters = n_iters
    converged = False

    while iters != 0 and not converged:
        #assign elements to new clusters    
        D = _dist_all(X, C)
        assign = D.argmin(axis=1)
        
        #check if converged, if not compute new centroids
        if prev_assign is not None and not (prev_assign - assign).any():
            converged = True
        else: 
            C_final = _compute_centroids(X, assign, num_clusters)

        prev_assign = assign
        iters -= 1
    
    return C_final, assign

def cost(X, C, assign):
    cost = 0
    for k in set(assign):
        idx = assign == k
        cost += _dist_all(X[idx], C[k][np.newaxis]).sum()
    return cost

def klkmeans(X, num_clusters, n_iters=-1, n_runs=10):

    min_cost = float('+inf')
    best_C = None
    best_assign = None

    for _ in xrange(n_runs):
        assign = np.random.randint(0, num_clusters, X.shape[0])
        C = _compute_centroids(X, assign, num_clusters)

        C, assign = _base_kmeans(X, C, n_iters)
        clust_cost = cost(X, C, assign)

        if clust_cost < min_cost:
            best_C = C
            best_assign = assign

    return best_C, best_assign

if __name__ == '__main__':
    np.seterr(all='raise')
    X = np.zeros((200, 1000))
    X[0:100] = 1
    X[100:200, 500:] = 1
    X += 1e-20
    
    X = (X.T / X.sum(axis=1)).T
    C, assign = klkmeans(X, 2)
    
    assert ((C.sum(axis=1) - 1) < 1e-10).all()
    assert (assign[0:100] != assign[100:]).all()
    
    import os
    dir_ = os.path.dirname(__file__)
    fpath = os.path.join(dir_, 'testdata.dat')
    X = np.genfromtxt(fpath)

    C, assign = klkmeans(X, 5)
    assert ((C.sum(axis=1) - 1) < 1e-10).all()
    assert len(set(assign)) == 5
