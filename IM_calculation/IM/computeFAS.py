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


def get_konno_matrix(size):
    with matrix_lock:
        if size not in matrices.keys():
            print(f"Loading matrix of size {size-1}")
            matrices[size] = np.load(
                os.path.join(
                    os.path.dirname(__file__), "KO_matrices", f"KO_{size - 1}.npy"
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
    waveforms = waveform[:, :2]
    fa_spectrum, fa_frequencies = generate_fa_spectrum(waveforms, dt, waveform.shape[0])
    fa_spectrum = np.abs(fa_spectrum)

    # get appropriate konno ohmachi matrix
    konno = get_konno_matrix(len(fa_spectrum))

    # apply konno ohmachi smoothing
    fa_smooth = np.dot(fa_spectrum.T, konno).T

    # interpolate at output frequencies
    interpolator = interp1d(fa_frequencies, fa_smooth, axis=0, fill_value="extrapolate")
    return interpolator(fa_frequencies_int)
