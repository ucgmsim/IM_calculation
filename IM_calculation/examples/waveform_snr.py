from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import IM_calculation.IM.snr_calculation as snr_calc
from IM_calculation.IM.read_waveform import create_waveform_from_data

data_dir = Path(__file__).parent / "resources"

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
comp_000 = comp_000[~np.isnan(comp_000)]
comp_090 = comp_090[~np.isnan(comp_090)]
comp_ver = comp_ver[~np.isnan(comp_ver)]

# Create the waveform object
waveform = create_waveform_from_data(
    np.stack((comp_090, comp_000, comp_ver), axis=1), NT=NT, DT=DT
)

# Index of the start of the P-wave
tp = 1320

# Example of creating a common frequency vector
# common_freqs = np.logspace(np.log10(0.05), np.log10(50), num=100, base=10.0)

(snr, frequencies, fas_signal, fas_noise, Ds, Dn) = snr_calc.get_snr_from_waveform(
    waveform, tp, DT, common_frequency_vector=None
)

if snr is not None:
    # Plot the waveform
    n_comps = snr.shape[1]
    comp_names = ["090", "000", "ver"]
    fig, axs = plt.subplots(n_comps, 2, figsize=(18, 12))
    for i in range(0, n_comps):
        # Waveform
        axs[i, 0].set_title(f"{comp_names[i]} Waveform")
        axs[i, 0].set_xlabel("Time (s)")
        axs[i, 0].set_ylabel("Acceleration (g)")
        axs[i, 0].plot(waveform.times, waveform.values.T[i])
        if tp is not None:
            axs[i, 0].axvline(waveform.times[tp], color="r", label="tp")
            axs[i, 0].legend()
        # SNR
        axs[i, 1].set_title(f"{comp_names[i]} SNR")
        axs[i, 1].set_xlabel("Frequency (Hz)")
        axs[i, 1].set_ylabel("SNR")
        axs[i, 1].plot(frequencies, snr[:, i])
        axs[i, 1].loglog()

    fig.tight_layout(pad=5.0)
    plt.show()
