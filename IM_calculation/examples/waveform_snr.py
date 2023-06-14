from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import IM_calculation.IM.snr_calculation as snr_calc
from IM_calculation.IM.read_waveform import create_waveform_from_data

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
comp_ver = pd.read_csv(
    data_dir / "DSZ.ver", sep="\s+", header=None, skiprows=2
).values.ravel()

# Remove the nan-values on the last line
# due to how the file was, generally shouldn't
# be needed
# comp_000 = comp_000[~np.isnan(comp_000)]
comp_090 = comp_090[~np.isnan(comp_090)]
# comp_ver = comp_ver[~np.isnan(comp_ver)]

# Create the waveform object
# waveform = create_waveform_from_data(
#     np.stack((comp_090, comp_000, comp_ver), axis=1), NT=NT, DT=DT
# )
waveform = create_waveform_from_data(
    np.stack((comp_090,), axis=1), NT=NT, DT=DT
)

# Index of the start of the P-wave
tp = 1320

common_freqs = np.logspace(np.log10(0.05), np.log10(50), num=100, base=10.0)

snr, Ds, Dn = snr_calc.get_snr_from_waveform(waveform, tp, common_frequency_vector=common_freqs)

# Plot the waveform
n_comps = snr.shape[1]
comp_names = ["090", "000", "ver"]
fig, axs = plt.subplots(n_comps, 2, figsize=(18, 12))
for i in range(0, n_comps):
    axs[0].set_title(f"{comp_names[i]} Waveform")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Acceleration (g)")
    axs[0].plot(waveform.times, waveform.values.T[i])
    if tp is not None:
        axs[0].axvline(waveform.times[tp], color="r", label="tp")
        axs[0].legend()
    axs[1].set_title(f"{comp_names[i]} SNR")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("SNR")
    axs[1].plot(common_freqs, snr[:, i])
    axs[1].loglog()

plt.legend()
fig.tight_layout(pad=5.0)
plt.show()
print("Yay")