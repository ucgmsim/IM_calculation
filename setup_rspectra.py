"""
This should only be used for installing the cython code locally for the unit tests,
in all other scenarios IM_calculation should be installed as a package using
setup.py (which will builds the cython code automatically)
command: python setup_rspectra.py build_ext --inplace
"""
from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize
from Cython.Distutils import build_ext



setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=[Extension("IM_calculation.IM.rspectra_calculations.rspectra",
                           ["IM_calculation/IM/rspectra_calculations/rspectra.pyx"])],
    # ext_modules=cythonize("IM_calculation/IM/rspectra_calculations/rspectra.pyx"),
    include_dirs=[numpy.get_include()],
)