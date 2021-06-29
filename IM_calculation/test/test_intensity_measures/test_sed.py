"""
Specific Energy Density IM Tests.
"""

from glob import glob
import os

import numpy as np
import pytest

from IM_calculation.IM.intensity_measures import get_specific_energy_density_nd
from qcore.timeseries import acc2vel, read_ascii

waveforms = [
    path
    for path in glob(
        os.path.join(os.path.dirname(__file__), os.pardir, "waveforms", "*", "*")
    )
    if os.path.splitext(path)[1] in [".000", ".090", ".ver"]
]
components = list(map(os.path.basename, waveforms))
# results taken from seismo signal
results = {
    "HALS.090": 0.81750,
    "HALS.000": 1.51869,
    "HALS.ver": 0.05813,
    "SEDS.090": 1180.01094,
    "SEDS.000": 1083.55882,
    "SEDS.ver": 64.72457,
    "WDFS.090": 206.22053,
    "WDFS.000": 247.10323,
    "WDFS.ver": 28.11681,
}


@pytest.mark.parametrize(
    "component_name, expected_im",
    list(results.items()),
)
def test_sed_im(component_name, expected_im):
    try:
        path = waveforms[components.index(component_name)]
    except ValueError:
        # missing test data
        print(f"Missing waveform data for {component_name}.")
        raise

    waveform, meta = read_ascii(path, meta=True)
    # waveform in g -> cm/s^2
    velocity = acc2vel(waveform * 980.665, dt=meta["dt"])
    times = np.arange(meta["nt"], dtype=np.float32)
    times *= meta["dt"]
    im = get_specific_energy_density_nd(velocity, times)

    assert np.isclose(im, expected_im, rtol=0.005)
