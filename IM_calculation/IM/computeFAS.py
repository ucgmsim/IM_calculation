#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:06:47 2019

@author: robin
"""
import os
from pathlib import Path
from threading import Lock
from typing import List

import numpy as np
from scipy.interpolate import interp1d
from threadpoolctl import threadpool_limits

matrices = {}
matrix_lock = Lock()

# Limit the number of threads used by numpy and scipy
threadpool_limits(limits=1, user_api='blas')


def get_konno_matrix(size: int, directory: Path = None):
    if directory is None:
        directory = Path(__file__).parent / "KO_matrices"
    with matrix_lock:
        if size not in matrices.keys():
            ko_matrix_file = directory / f"KO_{size - 1}.npy"
            if not ko_matrix_file.exists():
                num_to_gen = int((np.log(size - 1) / np.log(2)) - 9)
                raise FileNotFoundError(
                    f"KO matrix file {ko_matrix_file} not found, please make sure to run the "
                    f"A_KonnoMatricesComputation.py script with number to generate going upto at least {num_to_gen}"
                )
            matrices[size] = np.load(
                ko_matrix_file,
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
