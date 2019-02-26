"""
command: python setup_rspectra.py build_ext --inplace
"""

from distutils.core import setup

import numpy
from Cython.Build import cythonize


setup(
    ext_modules=cythonize("IM/rspectra_calculations/rspectra.pyx"),
    include_dirs=[numpy.get_include()]
)