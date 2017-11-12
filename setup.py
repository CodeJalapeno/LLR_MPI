from distutils.core import setup, Extension
import numpy.distutils.misc_util

module_LLR = Extension('LLR', sources=['Py_LLR_prox.c'],
include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs())

setup(ext_modules=[module_LLR], headers=headers)
