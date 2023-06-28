from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

import IM_calculation.IM.computeFAS as computeFAS
from IM_calculation.IM.read_waveform import Waveform


def apply_taper(acc: np.ndarray, percent: float = 0.05):
    """
    Applies a hanning taper to the start and end of the acceleration

    Parameters
    ----------
    acc : np.ndarray
        The acceleration values to apply the taper to,
        Can also be a 2D array of components,
        in which case the taper is applied to each column
    percent : float, optional
        The percentage of the waveform to taper, by default 0.05
    """
    npts = len(acc)
    ntap = int(npts * percent)
    hanning = np.hanning(ntap * 2 + 1)
    broadcasted_hanning = np.broadcast_to(
        hanning[:, np.newaxis], (len(hanning), acc.shape[1])
    )
    acc[:ntap] *= broadcasted_hanning[:ntap]
    acc[npts - ntap :] *= broadcasted_hanning[ntap + 1 :]
    return acc


def get_snr_from_waveform(
    waveform: Waveform,
    tp: float,
    ko_matrix_path: Path = None,
    common_frequency_vector: np.asarray = None,
):
    """
    Calculates the SNR of a waveform given a tp and common frequency vector

    Parameters
    ----------
    waveform : Waveform
        The waveform object to calculate the SNR for
        must have values and times attributes as well as DT defined
    tp : float
        The index of the p-arrival
    ko_matrix_path : Path, optional
        The path to the Ko matrices, by default None
    common_frequency_vector : np.asarray, optional
        The frequency vector to use for the SNR calculation,
        by default takes the frequencies from FAS
    """
    # Get the waveform values for comp 090 and times
    acc = waveform.values
    t = waveform.times

    # Calculate signal and noise areas
    signal_acc, noise_acc = acc.copy(), acc[:tp]
    signal_duration, noise_duration = t[-1], t[tp]

    # Ensure the noise is not shorter than 1s
    if noise_duration < 1:
        # Note down the ID of the waveform and ignore
        print(
            f"Waveform {waveform.station_name} has noise duration of {noise_duration}s"
        )
        return None, None, None, None, None, None

    # Add the tapering to the signal and noise
    taper_signal_acc = apply_taper(signal_acc)
    taper_noise_acc = apply_taper(noise_acc)

    # Generate FFT for the signal and noise
    fas_signal, frequency_signal = computeFAS.generate_fa_spectrum(
        taper_signal_acc, waveform.DT, len(taper_signal_acc)
    )
    fas_noise, frequency_noise = computeFAS.generate_fa_spectrum(
        taper_noise_acc, waveform.DT, len(taper_signal_acc)
    )

    # Take the absolute value of the FAS
    fas_signal = np.abs(fas_signal)
    fas_noise = np.abs(fas_noise)

    # Get appropriate konno ohmachi matrix
    konno_signal = computeFAS.get_konno_matrix(len(fas_signal), directory=ko_matrix_path)
    konno_noise = computeFAS.get_konno_matrix(len(fas_noise), directory=ko_matrix_path)

    # Apply konno ohmachi smoothing
    fa_smooth_signal = np.dot(fas_signal.T, konno_signal).T
    fa_smooth_noise = np.dot(fas_noise.T, konno_noise).T

    if common_frequency_vector is not None:
        # Interpolate SNR at common frequencies
        inter_signal_f = interp1d(
            frequency_signal, fa_smooth_signal, axis=0, fill_value="extrapolate"
        )
        inter_noise_f = interp1d(
            frequency_noise, fa_smooth_noise, axis=0, fill_value="extrapolate"
        )
        inter_signal = inter_signal_f(common_frequency_vector)
        inter_noise = inter_noise_f(common_frequency_vector)

        # Interpolate FAS at common frequencies
        inter_fas_signal_f = interp1d(
            frequency_signal, fas_signal, axis=0, fill_value="extrapolate"
        )
        inter_fas_noise_f = interp1d(
            frequency_noise, fas_noise, axis=0, fill_value="extrapolate"
        )
        fas_signal = inter_fas_signal_f(common_frequency_vector)
        fas_noise = inter_fas_noise_f(common_frequency_vector)
    else:
        inter_signal = fa_smooth_signal
        inter_noise = fa_smooth_noise

    # Calculate the SNR
    snr = (inter_signal / np.sqrt(signal_duration)) / (
        inter_noise / np.sqrt(noise_duration)
    )
    frequencies = (
        frequency_signal if common_frequency_vector is None else common_frequency_vector
    )

    return snr, frequencies, fas_signal, fas_noise, signal_duration, noise_duration
