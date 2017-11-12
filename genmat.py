import numpy as np
import scipy.io
import sys

DATA_OPT = int(sys.argv[1])

if DATA_OPT == 1:
    filename = '../Data/invivo_perfusion.mat'
    nb = 5
    coef = 0.05
elif DATA_OPT == 2:
    filename = '../Data/short_axis_cine.mat'
    nb = 5
    coef = 20
elif DATA_OPT == 3:
    filename = '../Data/long_axis_cine.mat'
    nb = 5
    coef = 1
elif DATA_OPT == 4:
    filename = '../Data/brain_t1_1.mat'
    nb = 5
    coef = 25
elif DATA_OPT == 5:
    filename = '../Data/brain_t1_2.mat'
    nb = 5
    coef = 10

mat = scipy.io.loadmat(filename)
f = mat['f']
(nx,ny,nc,nt) = f.shape[0:4];
max_iter = 50;
mu = 1.0/(nb*nb);
print nx
print ny
print nc
print nt
print nb
print max_iter
print coef
print mu

for i in range(nx):
    for j in range(ny):
        for k in range(nc):
            for l in range(nt):
                print "{0},{1}".format(f[i,j,k,l].real,f[i,j,k,l].imag)
