"""
Install using pip, e.g. pip install ./IM_Calculation
use --no-deps to prevent re-installation of dependencies
use -I to force re-install
"""
from pathlib import Path
from setuptools import find_packages, dist
from distutils.core import setup
from distutils.command.build_py import build_py
from distutils.extension import Extension

Distribution().fetch_build_eggs(['Cython', 'numpy>=1.14.3'])

import numpy
from Cython.Distutils import build_ext


class build_konno_matricies(build_py):
    """Post-installation for development mode."""

    def run(self):
        from IM_calculation.scripts.A_KonnoMatricesComputation import (
            createKonnoMatrices,
        )

        createKonnoMatrices(Path(__file__).parent / "IM_calculation" / "IM" / "KO_matrices")
        build_py.run(self)


setup(
    name="IM-calc",
    version="19.5.1",
    packages=find_packages(),
    url="https://github.com/ucgmsim/IM_calculation",
    description="IM calculation code",
    install_requires=["obspy", "pandas"],
    cmdclass={"build_ext": build_ext, "build_py": build_konno_matricies},
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
