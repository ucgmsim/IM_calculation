from pathlib import Path
import matplotlib.pyplot as plt
from typing import List

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import IM_calculation.IM.im_calculation as calc
import IM_calculation.IM.computeFAS as computeFAS
from qcore.constants import Components


def apply_taper(acc):
    npts = len(acc)
    ntap = int(npts * 0.05)
    hanning = np.hanning(ntap * 2 + 1)
    acc[:ntap] *= hanning[:ntap]
    acc[npts - ntap:] *= hanning[ntap + 1:]
    return acc


def get_snr_from_waveform(waveform, tp, common_frequency_vector: List[float] = np.logspace(0.05, 50, num=100, base=10.0)):
    # Get the waveform values for comp 090 and times
    acc = waveform.values.T[0]
    t = waveform.times

    # Calculate signal and noise areas
    signal_acc, noise_acc = acc.copy(), acc[:tp]
    signal_duration, noise_duration = t[-1], t[tp]

    # Ensure the noise is not shorter than 1s
    if noise_duration < 1:
        # Note down the ID of the waveform and ignore
        print(f"Waveform {waveform.station_name} has noise duration of {noise_duration}s")

    # Add the tapering to the signal and noise
    taper_signal_acc = apply_taper(signal_acc)
    taper_noise_acc = apply_taper(noise_acc)

    # Generate FFT for the signal and noise
    fas_signal, frequency_signal = computeFAS.generate_fa_spectrum(taper_signal_acc, waveform.DT, len(taper_signal_acc))
    fas_noise, frequency_noise = computeFAS.generate_fa_spectrum(taper_noise_acc, waveform.DT, len(taper_noise_acc))

    # Get appropriate konno ohmachi matrix
    konno_signal = computeFAS.get_konno_matrix(len(fas_signal))
    konno_noise = computeFAS.get_konno_matrix(len(fas_noise))

    # Apply konno ohmachi smoothing
    fa_smooth_signal = np.dot(fas_signal.T, konno_signal).T
    fa_smooth_noise = np.dot(fas_noise.T, konno_noise).T

    # Interpolate at common frequencies
    inter_signal_f = interp1d(frequency_signal, fa_smooth_signal, axis=0, fill_value="extrapolate")
    inter_noise_f = interp1d(frequency_noise, fa_smooth_noise, axis=0, fill_value="extrapolate")
    inter_signal = inter_signal_f(common_frequency_vector)
    inter_noise = inter_noise_f(common_frequency_vector)

    # Calculate the SNR
    snr = (inter_signal / np.sqrt(signal_duration)) / (inter_noise / np.sqrt(noise_duration))

    return snr, signal_duration, noise_duration
