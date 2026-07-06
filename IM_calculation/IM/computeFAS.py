#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:06:47 2019

@author: robin
"""
import os
from threading import Lock
from typing import List

import numpy as np
from scipy.interpolate import interp1d

matrices = {}
matrix_lock = Lock()


def get_konno_matrix(size: int, directory: str = None):
    if directory is None:
        directory = os.path.join(
                    os.path.dirname(__file__), "KO_matrices"
                )
    with matrix_lock:
        if size not in matrices.keys():
            matrices[size] = np.load(
                os.path.join(
                    directory, f"KO_{size - 1}.npy"
                ),
                mmap_mode="r",
            )
    return matrices[size]


def generate_fa_spectrum(y, dt, n):
    # RLL decided to put a more compact fft code here with proper normalization
    # Currently no time domain taper is applied
    nfft = 2 ** int(np.ceil(np.log2(n)))
    fa_spectrum = np.fft.rfft(y, n=nfft, axis=0) * dt
    fa_frequencies = np.fft.rfftfreq(nfft, dt)
    return fa_spectrum, fa_frequencies


def get_fourier_spectrum(
    waveform: np.ndarray,
    dt: float = 0.005,
    fa_frequencies_int: List[float] = np.logspace(-1, 2, num=100, base=10.0),
):
    fa_spectrum, fa_frequencies = generate_fa_spectrum(waveform, dt, waveform.shape[0])
    fa_spectrum = np.abs(fa_spectrum)

    # get appropriate konno ohmachi matrix
    konno = get_konno_matrix(len(fa_spectrum))

    # apply konno ohmachi smoothing
    fa_smooth = np.dot(fa_spectrum.T, konno).T

    # interpolate at output frequencies
    interpolator = interp1d(fa_frequencies, fa_smooth, axis=0, fill_value="extrapolate")
    return interpolator(fa_frequencies_int)


def get_fourier_spectrum_with_eas(waveform, dt, fa_frequencies_int, ko_directory=None):
    """Smoothed per-component FAS plus the NGA-West2 EAS.

    waveform: (nt, n_comp) acceleration; columns are [090, 000[, ver]].
    Returns (fas, eas):
      fas: (n_out, n_comp) Konno-Ohmachi-smoothed FAS per component, interpolated
           to fa_frequencies_int.
      eas: (n_out,) = interp(KOsmooth(sqrt(0.5*(raw090^2 + raw000^2)))) -- the two
           horizontals are combined on the RAW spectra, then smoothed. None if the
           waveform has fewer than two components.
    """
    fa_spectrum, fa_frequencies = generate_fa_spectrum(waveform, dt, waveform.shape[0])
    fa_spectrum = np.abs(fa_spectrum)                    # raw |rfft|*dt per component
    konno = get_konno_matrix(len(fa_spectrum), directory=ko_directory)

    fas_smooth = np.dot(fa_spectrum.T, konno).T          # smooth each component
    fas = interp1d(fa_frequencies, fas_smooth, axis=0, fill_value="extrapolate")(
        fa_frequencies_int
    )

    eas = None
    if fa_spectrum.shape[1] >= 2:
        eas_unsmoothed = np.sqrt(
            0.5 * (fa_spectrum[:, 0] ** 2 + fa_spectrum[:, 1] ** 2)
        )
        eas = interp1d(
            fa_frequencies, np.dot(eas_unsmoothed, konno), fill_value="extrapolate"
        )(fa_frequencies_int)
    return fas, eas
