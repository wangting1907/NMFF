# -*- coding: utf-8 -*-
"""
Created on Sat May 27 16:52:14 2023

@author: 19652
"""

import numpy as np
from scipy.optimize import nnls

def fNorm(X, Frac):
    elemFrac = X**Frac
    upperLimit = 100
    if Frac < 0:
        elemFrac[elemFrac > upperLimit] = upperLimit
    f = np.sum(elemFrac)
    return f


def FCLSU(HIM, M):
    nbands, nsamples = HIM.shape
    nbands, p = M.shape
    Delta = 1/1000
    N = np.zeros((nbands+1, p))
    N[:nbands, :p] = Delta * M
    N[nbands, :] = 1
    s = np.zeros(nbands+1)
    out = np.zeros((nsamples, p))
    for i in range(nsamples):
        s[:nbands] = Delta * HIM[:, i]
        s[nbands] = 1
        Abundances, _ = nnls(N, s)
        out[i, :] = Abundances
    return out

def hyperFcls(M, U):
    if U.ndim != 2:
        raise ValueError('M must be a p x q matrix.')

    p1, N = M.shape
    p2, q = U.shape
    if p1 != p2:
        raise ValueError('M and U must have the same number of spectral bands.')

    p = p1
    X = np.zeros((q, N))
    Mbckp = U.copy()

    for n1 in range(N):
        count = q
        done = False
        ref = np.arange(q)
        r = M[:, n1].reshape(-1, 1)
        U = Mbckp.copy()
        alpha = np.zeros((q, 1))

        while not done:
            als_hat = np.linalg.inv(U.T @ U) @ U.T @ r
            s = np.linalg.inv(U.T @ U) @ np.ones((count, 1))

            afcls_hat = als_hat - np.linalg.inv(U.T @ U) @ np.ones((count, 1)) @ np.linalg.inv(np.ones((1, count)) @ np.linalg.inv(U.T @ U) @ np.ones((count, 1))) @ (np.ones((1, count)) @ als_hat - 1)

            if np.sum(afcls_hat > 0) == count:
                alpha[ref] = afcls_hat
                break
            idx = np.where(afcls_hat < 0)[0]
            afcls_hat[idx] = afcls_hat[idx] / s[idx]
            maxIdx = idx[np.argmax(np.abs(afcls_hat[idx]))]
            alpha[maxIdx] = 0
            keep = np.setdiff1d(np.arange(U.shape[1]), maxIdx)
            U = U[:, keep]
            count = count - 1
            ref = ref[keep]

        X[:, n1] = alpha.flatten()

    return X

def nmfAbundance1(V, N, H_I, alpha, tolerance, maxIter):
    # this function use nmf-similiar method to factorization hyperspectrum
    # W,H:          output----factored output for hyperspectral data
    # V:            input ----original hyperspectroal
    # N:            input ----endmember numbers
    # H_I:          H initial value
    # alpha:        input ----scale factor 
    # tolerance:    input ----input for error tolerance
    # maxIter:      input ----input for max iteration times

    row_V, col_V = V.shape
    V_EXT = np.concatenate((alpha*V, np.ones((row_V, 1))), axis=1)

    V_cut = V_EXT[0:row_V, :]
    row_V, col_V = V_cut.shape
    W = np.abs(np.random.randn(row_V, N))
    sum_W = np.sum(W, axis=1)
    for i in range(row_V):
        W[i, :] = W[i, :] / sum_W[i]
    #W=W_init
    H = np.concatenate((H_I, np.ones((N, 1))), axis=1) # add sum is 1 to the equation
    
    j = 1 # record iter
    
    # record error for every iteration
    E = np.zeros(maxIter)
    # get the error
    e = V_cut - np.dot(W, H)
    e2 = np.sum(e**2)
    Iter_n = 1 # iteration times

    while e2 > tolerance:
        if Iter_n == maxIter:
            break
        
        # get next W
        VH = np.dot(V_cut, H.T)
        WHH = np.dot(W, np.dot(H, H.T))
        W = W * (VH / WHH) # update W
        
        # get error
        e = V_cut - np.dot(W, H)
        e2 = np.sum(e**2)
        
        j += 1
        E[Iter_n] = e2
        
        disp_str = f'[{Iter_n}] Loss: {e2}'
        #print(disp_str)
        
        Iter_n += 1
    
    E = E[0:Iter_n]
    
    return W



































def nmfAbundance(V, N, H_I, W_init, alpha, tolerance, maxIter):
    # this function use nmf-similiar method to factorization hyperspectrum
    # W,H:          output----factored output for hyperspectral data
    # V:            input ----original hyperspectroal
    # N:            input ----endmember numbers
    # H_I:          H initial value
    # alpha:        input ----scale factor 
    # tolerance:    input ----input for error tolerance
    # maxIter:      input ----input for max iteration times

    row_V, col_V = V.shape
    V_EXT = np.concatenate((alpha*V, np.ones((row_V, 1))), axis=1)

    V_cut = V_EXT[0:row_V, :]
    row_V, col_V = V_cut.shape
    #W = np.abs(np.random.randn(row_V, N))
    #sum_W = np.sum(W, axis=1)
    #for i in range(row_V):
     #   W[i, :] = W[i, :] / sum_W[i]
    W=W_init
    H = np.concatenate((H_I, np.ones((N, 1))), axis=1) # add sum is 1 to the equation
    
    j = 1 # record iter
    
    # record error for every iteration
    E = np.zeros(maxIter)
    # get the error
    e = V_cut - np.dot(W, H)
    e2 = np.sum(e**2)
    Iter_n = 1 # iteration times

    while e2 > tolerance:
        if Iter_n == maxIter:
            break
        
        # get next W
        VH = np.dot(V_cut, H.T)
        WHH = np.dot(W, np.dot(H, H.T))
        W = W * (VH / WHH) # update W
        
        # get error
        e = V_cut - np.dot(W, H)
        e2 = np.sum(e**2)
        
        j += 1
        E[Iter_n] = e2
        
        disp_str = f'[{Iter_n}] Loss: {e2}'
        #print(disp_str)
        
        Iter_n += 1
    
    E = E[0:Iter_n]
    
    return W




def hyperNmfASCL1_2(X, AInit, SInit, tolObj, maxIter, fDelta):
    # Input:
    #     X: Hyperspectral data matrix (bandNum * sampleSize).
    #     AInit: Endmember initial matrix (bandNum * emNum).
    #     SInit: Abundance initial matrix (emNum * sampleSize).
    #     tolObj: Stop condition for the difference of the objective function between two iterations.
    #     maxIter: Maximum number of iterations.
    #     fDelta: Factor that controls the strength of the sum-to-one constraint.
    #
    # Output:
    #     A: Resultant endmember matrix (bandNum * emNum).
    #     S: Resultant abundance matrix (emNum * sampleSize).
    #     ARc: Iterative record for the endmember matrix (emNum * iterNum * bandNum).
    #     errRc: Iterative record for the error (iterNum).
    #     objRc: Iterative record for the objective value (iterNum).

    # Estimate the number of emNum endmembers using the HySime algorithm.
    # Currently, we omit this operation since we already know the number of endmembers in synthetic data.
    emNum = AInit.shape[1]

    # Estimate the weight parameter fLamda according to the sparsity measure over X.
    bandNum = X.shape[0]
    sampleNum = X.shape[1]
    sqrtSampleNum = np.sqrt(sampleNum)
    tmp = 0
    for l in range(bandNum):
        xl = X[l, :]
        tmp += (sqrtSampleNum - (np.linalg.norm(xl, 1) / np.linalg.norm(xl, 2))) / (sqrtSampleNum - 1)
    fLamda = tmp / np.sqrt(bandNum)

    # Record iteration.
    errRc = np.zeros(maxIter)
    objRc = np.zeros(maxIter)
    ARecord = np.zeros((emNum, maxIter, bandNum))

    # fLamda should be rescaled to the level of spectral sample value
    fLamda = fLamda 
    #fLamda = 0.1
    # Initialize A and S by randomly selecting entries in the interval [0, 1].
    # Rescale each column of S to unit norm.
    A = AInit
    S = SInit
    iterNum = 1

    # Run iterations.
    Xf = np.vstack((X, fDelta * np.ones((1, sampleNum))))
    Af = np.vstack((A, fDelta * np.ones((1, emNum))))
    err = 0.5 * np.linalg.norm(Xf[:bandNum, :] - np.dot(Af[:bandNum, :], S), ord=2) ** 2
    newObj = err + fLamda * fNorm(S, 1 / 2)
    oldObj = 0
    dispStr = 'Iteration {}, loss = {}'.format(iterNum, newObj)
    print(dispStr)

    # record iteration.
    errRc[iterNum-1] = err
    objRc[iterNum-1] = newObj
    for i in range(emNum):
        ARecord[i, iterNum-1, :] = A[:bandNum, i]

    while err > tolObj and iterNum < maxIter:
        oldObj = newObj
        # update A
        #A = Af * (Xf @ S.T) / (Af @ S @ S.T)
        A = Af
        # update S
        lowLimit = 0.01
        S[S < lowLimit] = lowLimit
        S1_2 = S**(-1/2)
        S = S * (A.T @ Xf) / (A.T @ A @ S + 0.5 * fLamda * S1_2)
        #Af = A
        err = 0.5 * np.linalg.norm(Xf[:bandNum, :] - np.dot(Af[:bandNum, :], S), ord=2) ** 2
        newObj = err + fLamda * fNorm(S, 1 / 2)

        iterNum += 1
        dispStr = 'Iteration {}, loss = {}'.format(iterNum, newObj)
        print(dispStr)

        # record iteration.
        errRc[iterNum-1] = err
        objRc[iterNum-1] = newObj
        for i in range(emNum):
            ARecord[i, iterNum-1, :] = A[:bandNum, i]

    ARc = ARecord[:, :iterNum, :]
    errRc = errRc[:iterNum]
    objRc = objRc[:iterNum]
    return S
   # return S, A, ARc, errRc, objRc