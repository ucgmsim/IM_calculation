import subprocess
import sys
from setuptools import find_packages, setup, Extension


# The following is required to build the Cython extension
# This ensures that Cython is installed before running the setup
try:
    from Cython.Distutils import build_ext
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Cython"])
    from Cython.Distutils import build_ext


setup(
    name="IM-calc",
    version="19.5.1",
    packages=find_packages(),
    url="https://github.com/ucgmsim/IM_calculation",
    description="IM calculation code",
    install_requires=["Cython", "obspy", "pandas"],
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
)
