from pathlib import Path

import pandas as pd
import numpy as np

import IM_calculation.IM.im_calculation as calc
from IM_calculation.IM.read_waveform import create_waveform_from_data
from qcore import constants

data_dir = Path(__file__).parent / "resources"

# IM Options
periods = np.logspace(np.log(0.1), np.log(10.0), 100)
fas_frequencies = np.logspace(-1, 2, num=100, base=10.0)

# Read the data
NT, DT = 11452, 0.01
comp_000 = pd.read_csv(
    data_dir / "DSZ.000", sep="\s+", header=None, skiprows=2
).values.ravel()
comp_090 = pd.read_csv(
    data_dir / "DSZ.090", sep="\s+", header=None, skiprows=2
).values.ravel()

# Remove the nan-values on the last line
# due to how the file was, generally shouldn't
# be needed
comp_000 = comp_000[~np.isnan(comp_000)]
comp_090 = comp_090[~np.isnan(comp_090)]

# Create the waveform object
waveform = create_waveform_from_data(
    np.stack((comp_090, comp_000), axis=1), NT=NT, DT=DT
)

# Calculate the IMs
im_options = {
    "pSA": periods,
    "SDI": periods,
    "FAS": calc.validate_fas_frequency(fas_frequencies),
}
im_results = calc.compute_measure_single(
    (waveform, None),
    ["PGA", "PGV", "pSA", "CAV", "FAS", "MMI", "AI", "Ds575", "Ds595"],
    [constants.Components.crotd50],
    im_options,
    [constants.Components.c090, constants.Components.c000],
)
