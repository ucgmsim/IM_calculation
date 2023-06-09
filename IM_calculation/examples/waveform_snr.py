from pathlib import Path
import matplotlib.pyplot as plt
from typing import List

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import IM_calculation.IM.sc
from qcore.constants import Components

# Input parameters
bb_file = Path("/home/joel/local/Cybershake/Hanmer_BB/Hanmer_REL01_BB.bin")
output = Path("/home/joel/local/Cybershake/Hanmer_IM")

# Define im_calculation parameters
file_type = "binary"

bbseries, station_names = calc.get_bbseis(bb_file, file_type, selected_stations=None, real_only=True)
comps = [Components.from_str(c) for c in bbseries.COMP_NAME.values()]

waveforms = calc.read_waveform.read_binary_file(
    bbseries, comps, station_names=station_names, wave_type=None, file_type=None, units="g"
)

waveform = waveforms[0][0]
tp = 2900  # Index of the start of the P-wave
# plot_waveform(waveform, tp=tp)

snr, Ds, Dn = get_snr_from_waveform(waveform, tp)