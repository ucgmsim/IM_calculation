from pathlib import Path
import matplotlib.pyplot as plt
from typing import List

import numpy as np
from scipy.interpolate import interp1d

import IM_calculation.IM.im_calculation as calc
import IM_calculation.IM.snr_calculation as snr_calc
import IM_calculation.IM.computeFAS as computeFAS
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


def plot_waveform(waveform, comps=None, tp=None):
    if comps is None:
        comps = [0]
    plt.close()
    # Create rows for each component in the values array
    fig, axs = plt.subplots(len(comps))
    plt.title(f"{waveform.station_name} WaveType={waveform.wave_type}")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (g)")
    if len(comps) == 1:
        axs.plot(waveform.times, waveform.values.T[comps[0]])
    else:
        for i in range(0, len(comps)):
            axs[i].plot(waveform.times, waveform.values.T[i])
    if tp is not None:
        if len(comps) == 1:
            axs.axvline(waveform.times[tp], color="r", label="tp")
        else:
            axs[0].axvline(waveform.times[tp], color="r", label="tp")
    plt.legend()
    plt.show()


# def apply_taper(acc):
#     npts = len(acc)
#     ntap = int(npts * 0.05)
#     hanning = np.hanning(ntap * 2 + 1)
#     acc[:ntap] *= hanning[:ntap]
#     acc[npts - ntap:] *= hanning[ntap + 1:]
#     return acc


# def get_snr_from_waveform(waveform, tp, common_frequency_vector: List[float] = np.logspace(0.05, 50, num=100, base=10.0)):
#     # Get the waveform values for comp 090 and times
#     acc = waveform.values.T[0]
#     t = waveform.times
#
#     # Calculate signal and noise areas
#     signal_acc, noise_acc = acc.copy(), acc[:tp]
#     signal_duration, noise_duration = t[-1], t[tp]
#
#     # Ensure the noise is not shorter than 1s
#     if noise_duration < 1:
#         # Note down the ID of the waveform and ignore
#         print(f"Waveform {waveform.station_name} has noise duration of {noise_duration}s")
#
#     # Add the tapering to the signal and noise
#     taper_signal_acc = apply_taper(signal_acc)
#     taper_noise_acc = apply_taper(noise_acc)
#
#     # Generate FFT for the signal and noise
#     fas_signal, frequency_signal = computeFAS.generate_fa_spectrum(taper_signal_acc, waveform.DT, len(taper_signal_acc))
#     fas_noise, frequency_noise = computeFAS.generate_fa_spectrum(taper_noise_acc, waveform.DT, len(taper_noise_acc))
#
#     # Get appropriate konno ohmachi matrix
#     konno_signal = computeFAS.get_konno_matrix(len(fas_signal))
#     konno_noise = computeFAS.get_konno_matrix(len(fas_noise))
#
#     # Apply konno ohmachi smoothing
#     fa_smooth_signal = np.dot(fas_signal.T, konno_signal).T
#     fa_smooth_noise = np.dot(fas_noise.T, konno_noise).T
#
#     # Interpolate at common frequencies
#     inter_signal_f = interp1d(frequency_signal, fa_smooth_signal, axis=0, fill_value="extrapolate")
#     inter_noise_f = interp1d(frequency_noise, fa_smooth_noise, axis=0, fill_value="extrapolate")
#     inter_signal = inter_signal_f(common_frequency_vector)
#     inter_noise = inter_noise_f(common_frequency_vector)
#
#     # Calculate the SNR
#     snr = (inter_signal / np.sqrt(signal_duration)) / (inter_noise / np.sqrt(noise_duration))
#
#     return snr, signal_duration, noise_duration


waveform = waveforms[0][0]
tp = 2900  # Index of the start of the P-wave
# plot_waveform(waveform, tp=tp)

snr, Ds, Dn = snr_calc.get_snr_from_waveform(waveform, tp)



# plt.close()
# plt.plot(t[:tp], noise_acc, label="Original Signal")
# plt.plot(t[:tp], taper_noise_acc, label="Taper Signal")
# plt.show()
# plt.close()
# Plot the waveforms times and values using matplotlib
# for site_wavetypes in waveforms:
#     for waveform in site_wavetypes:
#         # Create 3 rows for each component in the values array
#         fig, axs = plt.subplots(3)
#         plt.title(f"{waveform.station_name} WaveType={waveform.wave_type}")
#         plt.xlabel("Time (s)")
#         plt.ylabel("Acceleration (g)")
#         for i in range(0,3):
#             # axs[i].title(f"{list(bbseries.COMP_NAME.values())[i]}")
#             axs[i].plot(waveform.times, waveform.values.T[i])
#         plt.show()
#         plt.close()