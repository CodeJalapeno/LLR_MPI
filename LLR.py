import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double, c_void_p

# input type for the denoise function
# must be a double array, with single dimension that is contiguous
array_mri_double = npct.ndpointer(dtype=np.double, ndim=4, flags='CONTIGUOUS')

# load the library, using numpy mechanisms
libLLR = npct.load_library("libLLR", ".")

# setup the return types and argument types
libLLR.denoise.restype = None
libLLR.denoise.argtypes = [array_mri_double, array_mri_double, c_int, c_int,
    c_int, c_int, c_int, c_double, c_int, c_double]


def denoise(in_array, B, coef=0.2, max_iter=100, mu=None):
    if in_array.ndim != 4:
        raise Exception("Wrong input dimension!")
    out_array = np.zeros(in_array.shape)
    if mu is None:
        mu = 1.0/(B*B)
    libLLR.denoise(in_array, out_array, in_array.shape[0],
    in_array.shape[1], in_array.shape[2], in_array.shape[3], B, coef,
    max_iter, mu)
    return out_array
