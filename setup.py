"""
Install using pip, e.g. pip install ./IM_Calculation
use --no-deps to prevent re-installation of dependencies
use -I to force re-install
"""
from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension


setup(
    name="IM-calc",
    version="19.5.1",
    packages=find_packages(),
    url="https://github.com/ucgmsim/IM_calculation",
    description="IM calculation code",
    install_requires=["numpy>=1.14.3", "Cython", "obspy", "pandas"],
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
)
