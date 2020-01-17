import os
import pickle

import numpy as np
from qcore.timeseries import BBSeis

from IM_calculation.IM import computeFAS
from IM_calculation.test.test_common_set_up import INPUT, OUTPUT, set_up


def test_compute_FAS(set_up):
    function = "get_fourier_spectrum"
    for root_path in set_up:
        bb = BBSeis(os.path.join(root_path, INPUT, "BB.bin"))
        station = bb.stations[0].name
        waveform = bb.acc(station)
        test_output = computeFAS.get_fourier_spectrum(waveform, bb.dt)

        with open(
            os.path.join(root_path, OUTPUT, function + "_ret_val.P"), "rb"
        ) as load_file:
            bench_output = pickle.load(load_file)[:, :2]

        assert np.isclose(test_output, bench_output).all()


def test_generate_fa_spectrum(set_up):
    function = "generate_fa_spectrum"
    for root_path in set_up:
        bb = BBSeis(os.path.join(root_path, INPUT, "BB.bin"))
        station = bb.stations[0].name
        waveform = bb.acc(station)[:100]
        test_fa_spectrum, test_fa_frequencies = computeFAS.generate_fa_spectrum(waveform[:, :2], bb.dt, waveform.shape[0])

        with open(
            os.path.join(root_path, OUTPUT, function + "_ret_val.P"), "rb"
        ) as load_file:
            bench_fa_spectrum, bench_fa_frequencies = pickle.load(load_file)

        assert np.isclose(test_fa_spectrum, bench_fa_spectrum).all()
        assert np.isclose(test_fa_frequencies, bench_fa_frequencies).all()
