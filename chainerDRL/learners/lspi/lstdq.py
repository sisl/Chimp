#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: LSTDQ.PY
Date: Wednesday, December 16 2009
Description: LSTDQ implementation from Lagoudakis and Parr. 2003. Least-Squares Policy Iteration. Journal of Machine Learning Research.
"""

import os, sys, getopt, pdb, string

import numpy as np
import numpy.random as npr
import random as pr
import numpy.linalg as la
from multiprocessing import Pool, Queue, Process, log_to_stderr, SUBDEBUG, cpu_count
from utils import sp_create,chunk,sp_create_dict
import sys
import time
from trackknown import TrackKnown

import scipy.sparse as sp
import scipy.sparse.linalg as spla

#logger = log_to_stderr()
#logger.setLevel(SUBDEBUG)

def allclose(A,B):
    if sp.issparse(A):
        A = A.toarray()
    if sp.issparse(B):
        B = B.toarray()

    return np.allclose(A,B.reshape(A.shape))

def compare(method1, method2, *args):
    """
    Run two methods side by side and compare the results.
    """

    A,b,x,info1 = method1(*args)
    C,d,y,info2 = method2(*args)

    # Note: if testing Opt method may need to create a wrapper method that inverts B prior to the method call
    if not allclose(A,C):
        print "***** (A,C) are not CLOSE! *****"
    else:
        print "(A,C) are close!"

    if not allclose(b,d):
        print "****** (b,d) are not CLOSE! *****"
    else:
        print "(b,d) are close!"

    if not allclose(x, y):
        print "****** (x,y) are not CLOSE! *****"
    else:
        print "(x,y) are close!"

    # dump stuff out here if needed

def solve(A, b, method="pinv"):
    info = {}
    w = None

    if method == "pinv":
        info['acond'] = la.cond(A)
        w = np.dot(la.pinv(A),b)

    elif method == "dot":
        if sp.issparse(A):
            w = A.dot(b).toarray()[:,0]
        else:
            w = A.dot(b)
        
    elif method == "lsqr":
        squeeze_b = np.array(b.todense()).squeeze()
        qr_result = spla.lsqr(A,squeeze_b.T, atol=1e-8, btol=1e-8, show=True)
    
        # diagnostics are already computed so we always populate info in this case    
        w = qr_result[0]
        info['istop'] = qr_result[1]
        info['itn'] = qr_result[2]
        info['r1norm'] = qr_result[3]
        info['r2norm'] = qr_result[4]
        info['anorm'] = qr_result[5]
        info['acond'] = qr_result[6]
        info['xnorm'] = qr_result[7]
        info['var'] = qr_result[8]

    elif method == "spsolve":
        qr_result = spla.spsolve(A,b)
        w = qr_result
    
    else:
        raise ValueError, "Unknown solution method!"

    return w,info

def numpy_loop(D,env,w,damping=0.001):
    """
    Problematic if the number of features is very large.
    """
    k = len(w)
    A = np.eye(k) * damping
    b = np.zeros(k)

    for (s,a,r,ns,na) in D:
        features = env.phi(s,a)
        next = env.linear_policy(w, ns)
        newfeatures = env.phi(ns, next)

        A = A + np.outer(features, features - env.gamma * newfeatures)
        b = b + features * r

    return A,b

def dict_loop(D,env,w,damping=0.001):
    """
    Speedy and memory efficient.
    """
    k = len(w)
    A = {(i,i) : damping for i in xrange(k)}
    b = {}

    for (s,a,r,ns,na) in D:
        features = env.phi(s, a, sparse=True, format='rawdict')
        next = env.linear_policy(w, ns)
        newfeatures = env.phi(ns, next, sparse=True, format='rawdict')

        # for 1-dim array on vals, rows matter
        nf = features.copy()
        for i,v in newfeatures.items():
            nf[i] = nf.get(i,0) - env.gamma * v

        for i,v1 in features.items():
            for j,v2 in nf.items():
                A[i,j] = A.get((i,j), 0) +  v1 * v2
            b[i] = b.get(i,0) + v1 * r

    # convert to sparse matrices since these could be large
    A = sp_create_dict(A,k,k,format='csr')
    b = sp_create_dict(b,k,1,format='csr')
    return A,b

def sparse_loop(D,env,w,damping=0.001):
    """
    This is somewhat surprisingly the slowest.
    """
    k = len(w)
    A = sp.identity(k,format='csr') * damping
    b = sp_create(k,1,'csr')

    for (s,a,r,ns,na) in D:
        features = env.phi(s, a, sparse=True, format='csr')
        next = env.linear_policy(w, ns)
        newfeatures = env.phi(ns, next, sparse=True, format='csr')

        nf = features - env.gamma * newfeatures
        T = sp.kron(features, nf.T)
        A = A + T
        b = b + features * r

    return A,b

def opt_loop(D,env,w,damping=0.001):
    """
    Computes inverse iteratively.
    """
    k = len(w)
    B = np.eye(k) * 1.0/damping
    b = np.zeros(k)

    for (s,a,r,ns,na) in D:
        features = env.phi(s,a)
        next = env.linear_policy(w, ns)
        newfeatures = env.phi(ns, next)

        nf = features - env.gamma * newfeatures
        uv = np.outer(features, nf)
        N = B.dot(uv).dot(B)
        d = 1 + nf.dot(B).dot(features)
        B = B - N / d
        b = b + features * r

    return B,b

def sopt_loop(D,env,w,damping=0.001):
    """
    Sparse matrix version that computes inverse iteratively.
    """
    k = len(w)
    B = sp.identity(k,format='csr') * 1.0/damping
    b = sp_create(k,1,'csr')

    for (s,a,r,ns,na) in D:
        features = env.phi(s,a,sparse = True, format='csr')
        next = env.linear_policy(w, ns)
        newfeatures = env.phi(ns, next, sparse = True, format='csr')

        nf = features - env.gamma * newfeatures
        uv = sp.kron(features,nf.T)
        N = B.dot(uv).dot(B)
        d = 1 + nf.T.dot(B).dot(features)[0,0]

        B = B - N / d
        b = b + features * r

    return B,b

def rmax_loop(D,env,w,damping=0.001,rmax=1.0):
    """
    Rmax loop with numpy matrices.
    """
    k = len(w)
    A = np.eye(k) * damping
    b = np.zeros(k)
    grmax = rmax / (1.0 - env.gamma)

    for (s,a,r,ns,na) in D:
        if D.known_pair(s,a) and D.known_state(ns):
            features = env.phi(s,a)
            next = env.linear_policy(w, ns)
            newfeatures = env.phi(ns, next)
            A = A + np.outer(features, features - env.gamma * newfeatures)
            b = b + features * r
        elif D.known_pair(s,a):
            features = env.phi(s,a)
            A = A + np.outer(features, features)
            b = b + features * (r + env.gamma * grmax)          
        else:
            features = env.phi(s,a)
            A = A + np.outer(features,features)
            b = b + features * grmax
        for una in D.unknown(s):
            features = env.phi(s,una)
            A = A + np.outer(features,features)
            b = b + features * grmax

    return A,b

def srmax_loop(D, env, w, damping=0.001, rmax = 1.0):
    """
    Sparse rmax loop.
    """
    k = len(w)
    A = sp.identity(k,format='csr') * damping
    b = sp_create(k,1,'csr')
    grmax = rmax / (1.0 - env.gamma)

    for (s,a,r,ns,na) in D:
        if D.known_pair(s,a) and D.known_state(ns):
            features = env.phi(s, a, sparse=True, format='csr')
            next = env.linear_policy(w, ns)
            newfeatures = env.phi(ns, next, sparse=True, format='csr')
            nf = features - env.gamma * newfeatures
            T = sp.kron(features, nf.T)
            A = A + T
            b = b + features * r 
        elif D.known_pair(s,a):
            features = env.phi(s, a, sparse=True, format='csr')
            T = sp.kron(features, features.T)
            A = A + T
            b = b + features * (r + env.gamma * grmax)
        else:            
            features = env.phi(s, a, sparse=True, format='csr')
            T = sp.kron(features, features.T)
            A = A + T
            b = b + features * grmax
        for una in D.unknown(s):
            features = env.phi(s, una, sparse=True, format='csr')
            T = sp.kron(features, features.T)
            A = A + T
            b = b + features * grmax

    return A,b

def drmax_loop(D, env, w, damping=0.001, rmax=1.0):
    """
    Dictionary rmax loop.
    """
    k = len(w)
    A = {(i,i) : damping for i in xrange(k)}
    b = {}
    grmax = rmax / (1.0 - env.gamma)


    for (s,a,r,ns,na) in D:
        if D.known_pair(s,a) and D.known_state(ns):
            features = env.phi(s, a, sparse=True, format='rawdict')
            next = env.linear_policy(w, ns)
            newfeatures = env.phi(ns, next, sparse=True, format='rawdict')

            nf = features.copy()
            for i,v in newfeatures.items():
                nf[i] = nf.get(i,0) - env.gamma * v

            for i,v1 in features.items():
                for j,v2 in nf.items():
                    A[i,j] = A.get((i,j), 0) +  v1 * v2
                b[i] = b.get(i,0) + v1 * r

        elif D.known_pair(s,a):
            features = env.phi(s, a, sparse=True, format='rawdict')
            for i,v1 in features.items():
                for j,v2 in features.items():
                    A[i,j] = A.get((i,j), 0) + v1 * v2
                b[i] = b.get(i,0) + v1 * (r + env.gamma * grmax)
        
        else:            
            features = env.phi(s, a, sparse=True, format='rawdict')
            for i,v1 in features.items():
                for j,v2 in features.items():
                    A[i,j] = A.get((i,j), 0) + v1 * v2
                b[i] = b.get(i,0) + v1 * grmax

        for una in D.unknown(s):
            features = env.phi(s, una, sparse=True, format='rawdict')
            for i,v1 in features.items():
                for j,v2 in features.items():
                    A[i,j] = A.get((i,j), 0) + v1 * v2
                b[i] = b.get(i,0) + v1 * grmax

    A = sp_create_dict(A,k,k,format='csr')
    b = sp_create_dict(b,k,1,format='csr')
    return A,b



def OptLSTDQ(D,env,w,damping=0.001):
    """
    Use paper's suggested optimization method.
    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation
    """
    B,b = opt_loop(D,env,w,damping)
    w,info = solve(B,b,method="dot")
    return B,b,w,info


def SparseOptLSTDQ(D,env,w,damping=0.001):
    """
    Use paper's suggested optimization method.
    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation
    """
    B,b = sopt_loop(D,env,w,damping)
    w,info = solve(B,b,method="dot")
    return B,b,w,info

def LSTDQ(D,env,w,damping=0.001):
    """
    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation
    damping : keeps the result relatively stable 
    """
    A,b = numpy_loop(D,env,w,damping)
    w,info = solve(A,b,method="pinv")
    return A,b,w,info

def FastLSTDQ(D,env,w,damping=0.001):
    """
    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation
    damping : keeps the result relatively stable 
    """
    A,b = dict_loop(D,env,w,damping)
    print "Feature Sparsity: ", env.get_sparsity()
    w,info = solve(A,b,method="spsolve")
    return A,b,w,info

def SparseLSTDQ(D,env,w,damping=0.001):
    """
    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation
    damping : keeps the result relatively stable 
    """
    A,b = sparse_loop(D,env,w,damping)
    w,info = solve(A,b,method="spsolve")
    return A,b,w,info

def LSTDQRmax(D, env, w, damping=0.001, rmax = 1.0):
    """
    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation
    damping : keeps the result relatively stable (solves some difficulties with oscillation if A is singular)
    rmax : the maximum reward
    """
    A,b = rmax_loop(D,env,w,damping,rmax)
    w,info = solve(A,b,method="pinv")
    return A,b,w,info

def SparseLSTDQRmax(D, env, w, damping=0.001, rmax = 1.0):
    """
    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation
    damping : keeps the result relatively stable (solves some difficulties with oscillation if A is singular)
    rmax : the maximum reward
    """
    A,b = srmax_loop(D,env,w,damping,rmax)
    w,info = solve(A,b,method="spsolve")
    return A,b,w,info

def FastLSTDQRmax(D, env, w, damping=0.001, rmax = 1.0):
    """
    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation
    damping : keeps the result relatively stable (solves some difficulties with oscillation if A is singular)
    rmax : the maximum reward
    """
    A,b = drmax_loop(D,env,w,damping,rmax)
    w,info = solve(A,b,method="spsolve")
    return A,b,w,info
  
def ParallelLSTDQRmax(D,env,w,damping=0.001,rmax=1.0,ncpus=None):
    """
    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation
    damping : keeps the result relatively stable (solves some difficulties with oscillation if A is singular)
    rmax : the maximum reward
    ncpus : the number of cpus to use
    """
    if ncpus:
        nprocess = ncpus
    else:
        nprocess = cpu_count()
    
    pool = Pool(nprocess)
    indx = chunk(len(D),nprocess)
    results = []
    for (i,j) in indx:
        r = pool.apply_async(drmax_loop,(D[i:j],env,w,0.0,rmax)) # note that damping needs to be zero here
        results.append(r)
        
    k = len(w)
    A = sp.identity(k,format='csr') * damping
    b = sp_create(k,1,'csr')
    for r in results:
        T,t = r.get()
        A = A + T
        b = b + t

    # close out the pool of workers
    pool.close()
    pool.join()

    w,info = solve(A,b,method="spsolve")
    return A,b,w,info

def ParallelLSTDQ(D,env,w,damping=0.001,ncpus=None):
    """
    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation
    damping : keeps the result relatively stable 
    ncpus : the number of cpus to use
    """

    if ncpus:
        nprocess = ncpus
    else:
        nprocess = cpu_count()

    pool = Pool(nprocess)
    indx = chunk(len(D),nprocess)
    results = []
    for (i,j) in indx:
        r = pool.apply_async(dict_loop,(D[i:j],env,w,0.0)) # note that damping needs to be zero here
        results.append(r)
        
    k = len(w)
    A = sp.identity(k,format='csr') * damping
    b = sp_create(k,1,'csr')
    for r in results:
        T,t = r.get()
        A = A + T
        b = b + t

    # close out the pool of workers
    pool.close()
    pool.join()

    w,info = solve(A,b,method="spsolve")
    return A,b,w,info

if __name__ == '__main__':

    from gridworld.gridworld8 import SparseGridworld8
    import cPickle as pickle

    gw = SparseGridworld8(nrows = 5, ncols = 5, endstates = [0], walls = [])
    t = pickle.load(open("/Users/stober/wrk/lspi/bin/rmax_trace.pck"))
    policy0 = np.zeros(gw.nfeatures())
    track = TrackKnown(t, gw.nstates, gw.nactions, 1)
    compare(LSTDQRmax, ParallelLSTDQRmax, track, gw, policy0)
 