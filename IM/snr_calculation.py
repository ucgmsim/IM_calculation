"""Waveform SNR calculation"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr

from IM import im_calculation, ims


class SNRResult(NamedTuple):
    """Result of an SNR calculation.

    Contains the signal-to-noise ratio (SNR) calculations along with the Fourier amplitude
    spectra (FAS) for both signal and noise components, and their respective durations.
    """

    snr_df: pd.DataFrame
    """DataFrame containing the calculated SNR values for each component (000, 090, ver).
    The index represents frequencies and columns represent the different components."""

    fas_signal_df: pd.DataFrame
    """DataFrame containing the Fourier amplitude spectra of the signal portion
    for each component (000, 090, ver). The index represents frequencies and
    columns represent the different components."""

    fas_noise_df: pd.DataFrame
    """DataFrame containing the Fourier amplitude spectra of the noise portion
    for each component (000, 090, ver). The index represents frequencies and
    columns represent the different components."""

    signal_duration: float
    """Duration of the signal portion in seconds, calculated as the number of
    samples in the signal multiplied by the sampling interval (dt)."""

    noise_duration: float
    """Duration of the noise portion in seconds, calculated as the number of
    samples in the noise multiplied by the sampling interval (dt)."""


def calculate_snr(
    waveform: np.ndarray,
    dt: float,
    tp: int,
    ko_directory: Path,
    frequencies: np.ndarray = im_calculation.DEFAULT_FREQUENCIES,
) -> SNRResult:
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
    ko_directory : Path
        The path to the directory containing the Konno-Ohmachi matrices
    frequencies : np.ndarray, optional
        The frequency vector to use for the SNR calculation,
        by default takes the frequencies from FAS

    Returns
    -------
    SNRResult
        The output of the SNR calculation. See `SNRResult` documentation
        for details of each component of the SNR calculation.

    Raises
    ------
    ValueError
        If the noise duration is less than 1s and so SNR can't be computed.
    """
    # This extra time is to ensure that when a taper is applied, the signal part of the waveform
    # is not affected by the tapering. The tapering is applied to the signal and noise separately.
    (_, _, nt) = waveform.shape
    tp_extra = (nt - tp) / 19
    # Round up the tp_extra to the nearest highest integer
    tp_extra = int(np.ceil(tp_extra))

    # Calculate signal and noise areas
    signal_duration, noise_duration = (nt - max(tp - tp_extra, 0), tp)

    signal_acc, noise_acc = (
        waveform[:, :, nt - signal_duration :],
        waveform[:, :, :noise_duration],
    )

    # Ensure the noise is not shorter than 1s, if not then skip the calculation
    if noise_duration < 1:
        raise ValueError("Noise duration is less than 1s")

    # Apply Taper
    taper_signal_acc = signal_acc * sp.signal.windows.tukey(
        signal_acc.shape[-1], alpha=0.05
    )
    taper_noise_acc = noise_acc * sp.signal.windows.tukey(
        noise_acc.shape[-1], alpha=0.05
    )

    # Ensure float 64 for the waveform
    taper_signal_acc = taper_signal_acc.astype(np.float64)
    taper_noise_acc = taper_noise_acc.astype(np.float64)

    # Generate FFT for the signal and noise
    fas_signal = ims.fourier_amplitude_spectra(
        taper_signal_acc, dt, frequencies, ko_directory
    )
    fas_noise = ims.fourier_amplitude_spectra(
        taper_noise_acc, dt, frequencies, ko_directory
    )

    # Calculate the SNR. Dataset arithmetic aligns on variable name, so this
    # produces a 5-variable (000/090/ver/geom/eas) dataset just like fas_signal
    # and fas_noise.
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = (fas_signal * noise_duration) / (fas_noise * signal_duration)

    snr_df = _component_frame(snr)
    fas_signal_df = _component_frame(fas_signal)
    fas_noise_df = _component_frame(fas_noise)

    return SNRResult(
        snr_df, fas_signal_df, fas_noise_df, signal_duration, noise_duration
    )


def _component_frame(dataset: xr.Dataset) -> pd.DataFrame:
    """Take the 000/090/ver components of a single-station FAS dataset as a
    frequency-indexed DataFrame.
    """
    return dataset[["000", "090", "ver"]].isel(station=0, drop=True).to_dataframe()
