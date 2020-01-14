#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:06:47 2019

@author: robin
"""
import argparse
import os
import time
from typing import List

import numpy as np

import matplotlib.pyplot as plt


def generate_fa_spectrum(y, dt, n):
    # RLL decided to put a more compact fft code here with proper normalization
    # Currently no time domain taper is applied
    nfft = 2 ** int(np.ceil(np.log2(n)))
    fa_spectrum = np.fft.rfft(y, n=nfft) * dt
    fa_frequencies = np.fft.rfftfreq(nfft, dt)
    return fa_spectrum, fa_frequencies


def readGP(loc, fname):
    """
    Convenience function for reading files in the Graves and Pitarka format
    """
    with open("/".join([loc, fname]), "r") as f:
        lines = f.readlines()

    data = [[float(val) for val in line.split()] for line in lines[2:]]

    data = np.concatenate(data)

    line1 = lines[1].split()
    num_pts = float(line1[0])
    dt = float(line1[1])
    shift = float(line1[4])

    return data, num_pts, dt, shift


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_waveform")


def get_fourier_spectrum(
    waveform: np.ndarray,
    dt: float = 0.005,
    fa_frequencies_int: List[float] = np.logspace(
        -1, 2, num=100, base=10.0
    ),
):
    returns = []
    for w in waveform.T:
        fa_spectrum, fa_frequencies = generate_fa_spectrum(w, dt, len(w))
        fa_spectrum = np.abs(fa_spectrum)

        # get appropriate konno ohmachi matrix
        size = len(fa_spectrum)
        konno = np.load(os.path.join(os.path.dirname(__file__), "KO_matrices", f"KO_{size-1}.npy"))

        # apply konno ohmachi smoothing
        fa_smooth = np.dot(konno, fa_spectrum)

        # interpolate at output frequencies
        returns.append(np.interp(fa_frequencies_int, fa_frequencies, fa_smooth))

    return np.asarray(returns).T


def main():
    t1 = time.time()

    data, npts, dt, shift = readGP(os.getcwd(), "ADCS.000")

    fa_spectrum, fa_frequencies = generate_fa_spectrum(data, dt, npts)
    fa_spectrum = np.abs(fa_spectrum)

    # get appropriate konno ohmachi matrix
    size = len(fa_spectrum)
    konno = np.load("./KO_matrices/KO_{}}.npy".format(size - 1))

    # apply konno ohmachi smoothing
    fa_smooth = np.dot(konno, fa_spectrum)

    # output frequency vector
    fa_frequencies_int = np.logspace(-1, 2, num=100, base=10.0)

    # interpolate at output frequencies
    fa_smooth_int = np.interp(fa_frequencies_int, fa_frequencies, fa_smooth)

    plt.figure(figsize=(6, 4))
    plt.loglog(fa_frequencies, fa_spectrum, c="b")
    plt.loglog(fa_frequencies, fa_smooth, c="g")
    plt.loglog(fa_frequencies_int, fa_smooth_int, c="k")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("FAS (g-s)")
    plt.grid(True, which="both")
    plt.savefig("test.png", dpi=200)

    t2 = time.time()
    print("\nFAS time: %.1f sec." % (t2 - t1))


if __name__ == "__main__":
    main()
