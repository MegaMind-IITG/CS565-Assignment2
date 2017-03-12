import numpy as np
from collections import Counter
import operator
import cPickle as pickle
import math
from scipy import linalg
from numpy.linalg import norm
from random import normalvariate
from math import sqrt
import sys
from matrix import multimatrix, itermatrix, random_m, iterate_results
import matrix

def svd_1d(A, epsilon=1e-10):
    ''' The one-dimensional SVD '''
 
    n, m = A.shape
    print A.shape
    x = np.random.randn(m)
    x = x/norm(x)

    lastV = None
    currentV = x

    print "Multiplying A' and A"
    
    # Matrix Multiplication using Multithreading     
    # X = multimatrix(A.T)
    # Y = multimatrix(A)
    # B = X * Y

    B = np.matmul(A.T, A)
 
    iterations = 0
    print "Entering while loop"
    while True:
        iterations += 1
        # print iterations
        lastV = currentV
        currentV = np.dot(B, lastV)
        currentV = currentV / norm(currentV)
        if abs(np.dot(currentV, lastV)) > 1 - epsilon:
            print("converged in {} iterations!".format(iterations))
            return currentV

def svd(A, epsilon=1e-10,k=50):
    #implementation of svd using the power function
    n, m = A.shape
    svdSoFar = []
    for i in range(min(m,k)):
        print("Value of i: %d" % (i))
        matrixFor1D = A.copy()
        temp = svdSoFar

        for singularValue, u, v in svdSoFar[:i]:
            matrixFor1D = matrixFor1D + (singularValue * np.outer(u, v))
        print "Entering svd_1d"
        v = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
        u_unnormalized = np.matmul(A, v)
        sigma = norm(u_unnormalized)  # next singular value
        u = u_unnormalized / sigma
        svdSoFar.append((sigma, u, v))
    # transform it into matrices of the right shape
    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
    return singularValues, us.T, vs
