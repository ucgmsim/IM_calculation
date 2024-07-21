"""
Install using pip, e.g. pip install ./IM_Calculation
"""
from Cython.Distutils import build_ext
from setuptools import setup, Extension


setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=[
        Extension(
            "IM_calculation.IM.rspectra_calculations.rspectra",
            ["IM_calculation/IM/rspectra_calculations/rspectra.pyx"],
        ),
    ],
)
