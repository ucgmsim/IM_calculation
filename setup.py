import numpy
from Cython.Distutils import build_ext
from setuptools import Extension, find_packages, setup

setup(
    packages=find_packages(),
    cmdclass={"build_ext": build_ext},
    ext_modules=[
        Extension(
            "IM_calculation.IM.rspectra_calculations.rspectra",
            ["IM_calculation/IM/rspectra_calculations/rspectra.pyx"],
            include_dirs=[numpy.get_include()]
        ),
    ],
)
