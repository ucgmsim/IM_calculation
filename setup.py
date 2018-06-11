"""
command: python setup.py build_ext --inplace
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
    ext_modules=cythonize("IM/rspectra_calculations/rspectra.pyx"),
    include_dirs=[numpy.get_include()]
)