#!/usr/bin/python -W all

import numpy as np
from numpy import linalg as la
import LLR

def denoise(in_array, B, coef=0.1, max_iter=100, mu=None):
    return

X = np.random.randn(128,128,2,10)

#print X
B = 5

Y = LLR.denoise(X, B)

#print Y
(nx,ny,nc,nt) = X.shape


#print Y[:B,:B,:,:]
#print Y[:B,:B,:,:].reshape(B*B*nc,nt)
#print la.linalg.norm(Y[:B,:B,:,:].reshape(B*B*nc,nt),2)
#[u,s,v] = la.svd(X[:B,:B,:,:].reshape(B*B*nc,nt),full_matrices=False)
#print np.dot(u,np.dot(np.diag(np.minimum(s,0.1)),v.T))
