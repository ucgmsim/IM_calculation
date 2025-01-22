import multiprocessing

import numpy as np
import scipy as sp

from IM import im_calculation, ims


def calculate_snr(
    waveform: np.ndarray,
    dt: float,
    tp: float,
    frequencies: np.asarray = im_calculation.DEFAULT_FREQUENCIES,
    cores: int = multiprocessing.cpu_count(),
    ko_bandwidth: int = 40,
    apply_taper: bool = True,
):
    """
    Calculates the SNR of a waveform given a tp and common frequency vector

    Parameters
    ----------
    waveform : np.ndarray
        Waveform data as a NumPy array.
    dt : float
        The sampling rate of the waveform
    tp : float
        The index of the p-arrival
    frequencies : np.asarray, optional
        The frequency vector to use for the SNR calculation,
        by default takes the frequencies from FAS
    cores : int, optional
        Number of cores to use for parallel processing in FAS calculations.
    ko_bandwidth : int, optional
        Bandwidth for the Konno-Ohmachi smoothing, by default 40.
    apply_taper : bool, optional
        Whether to apply a taper of 5% to the signal and noise, by default True.
    """
    # Calculate signal and noise areas
    signal_acc, noise_acc = waveform[:, tp:], waveform[:, :tp]
    signal_duration, noise_duration = signal_acc.shape[1] * dt, noise_acc.shape[1] * dt

    # Ensure the noise is not shorter than 1s, if not then skip the calculation
    if noise_duration < 1:
        raise ValueError("Noise duration is less than 1s")

    # Add the tapering to the signal and noise
    if apply_taper:
        taper_signal_acc = (
            sp.signal.windows.tukey(signal_acc.shape[1], alpha=0.05).reshape(-1, 1)
            * signal_acc[:]
        )
        taper_noise_acc = (
            sp.signal.windows.tukey(noise_acc.shape[1], alpha=0.05).reshape(-1, 1)
            * noise_acc[:]
        )
    else:
        taper_signal_acc = signal_acc
        taper_noise_acc = noise_acc

    # Ensure float 32 for the waveform
    taper_signal_acc = taper_signal_acc.astype(np.float32)
    taper_noise_acc = taper_noise_acc.astype(np.float32)

    # Generate FFT for the signal and noise
    fas_signal = ims.fourier_amplitude_spectra(
        taper_signal_acc, dt, frequencies, cores, ko_bandwidth
    )
    fas_noise = ims.fourier_amplitude_spectra(
        taper_noise_acc, dt, frequencies, cores, ko_bandwidth
    )

    # Set values to NaN if they are outside the bounds of sample rate / 2
    sample_rate = 1 / dt
    fas_signal[:, :, frequencies > sample_rate / 2] = np.nan
    fas_noise[:, :, frequencies > sample_rate / 2] = np.nan

    # Calculate the SNR
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = (fas_signal / np.sqrt(signal_duration)) / (
            fas_noise / np.sqrt(noise_duration)
        )

    # Create SNR DataFrame with 000, 090 and ver
    snr_df = snr.to_dataframe().unstack(level="component")
    snr_df.index = snr.coords["frequency"].values
    snr_df.columns = snr_df.columns.droplevel(0)
    snr_df = snr_df[["000", "090", "ver"]]

    # Create FAS noise and signal DataFrames with 000, 090 and ver
    fas_signal_df = fas_signal.to_dataframe().unstack(level="component")
    fas_signal_df.index = fas_signal.coords["frequency"].values
    fas_signal_df.columns = fas_signal_df.columns.droplevel(0)
    fas_signal_df = fas_signal_df[["000", "090", "ver"]]

    fas_noise_df = fas_noise.to_dataframe().unstack(level="component")
    fas_noise_df.index = fas_noise.coords["frequency"].values
    fas_noise_df.columns = fas_noise_df.columns.droplevel(0)
    fas_noise_df = fas_noise_df[["000", "090", "ver"]]

    return snr_df, fas_signal_df, fas_noise_df, signal_duration, noise_duration
