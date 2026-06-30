from setuptools import setup, find_packages, Extension
import numpy
from Cython.Distutils import build_ext

# ...

setup(
    name="IM-calc",
    version="19.5.1",
    packages=find_packages(),
    url="https://github.com/ucgmsim/IM_calculation",
    description="IM calculation code",
    install_requires=["numpy>=1.14.3", "Cython", "obspy", "pandas"],
    cmdclass={"build_ext": build_ext},
    package_data={"": ["*.yaml"]},
    ext_modules=[
        Extension(
            "IM_calculation.IM.rspectra_calculations.rspectra",
            ["IM_calculation/IM/rspectra_calculations/rspectra.pyx"],
        ),
    ],
    scripts=[
        "IM_calculation/scripts/calculate_ims.py",
        "IM_calculation/scripts/calculate_rrups_single.py",
    ],
    include_dirs=[numpy.get_include()],
)
