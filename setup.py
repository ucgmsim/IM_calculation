# TODO: move setup.py outof Cython folder, otherwiare, .so file will be created under Cython/Cython/
"""
command: python setup.py build_ext --inplace
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("rspectra.pyx"),
    include_dirs=[numpy.get_include()]
)