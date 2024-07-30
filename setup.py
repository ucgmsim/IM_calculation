from setuptools import find_packages, setup, Extension
from Cython.Distutils import build_ext


setup(
    packages=find_packages(),
    cmdclass={"build_ext": build_ext},
    ext_modules=[
        Extension(
            "IM_calculation.IM.rspectra_calculations.rspectra",
            ["IM_calculation/IM/rspectra_calculations/rspectra.pyx"],
        ),
    ],
)
