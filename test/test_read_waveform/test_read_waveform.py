import os
import pickle

import numpy as np

from IM import read_waveform
from test.test_common_set_up import (
    INPUT,
    OUTPUT,
    set_up,
    compare_waveforms,
    compare_iterable
)


def get_common_waveform_values(root_path, function_name):
    with open(
        os.path.join(root_path, INPUT, function_name + "_bbseis.P"), "rb"
    ) as load_file:
        bbseis = pickle.load(load_file)
    with open(
        os.path.join(root_path, INPUT, function_name + "_comp.P"), "rb"
    ) as load_file:
        comp = pickle.load(load_file)
    with open(
        os.path.join(root_path, INPUT, function_name + "_wave_type.P"), "rb"
    ) as load_file:
        wave_type = pickle.load(load_file)
    with open(
        os.path.join(root_path, INPUT, function_name + "_file_type.P"), "rb"
    ) as load_file:
        file_type = pickle.load(load_file)

    bbseis.path = os.path.join(root_path, INPUT, 'BB.bin')

    return bbseis, comp, wave_type, file_type


def get_common_bbseis_values(root_path, function_name):
    with open(
        os.path.join(root_path, INPUT, function_name + "_station_names.P"), "rb"
    ) as load_file:
        station_names = pickle.load(load_file)

    with open(
        os.path.join(root_path, INPUT, function_name + "_units.P"), "rb"
    ) as load_file:
        units = pickle.load(load_file)
    return station_names, units


def test_calculate_timesteps(set_up):
    function = "calculate_timesteps"
    for root_path in set_up:
        with open(
            os.path.join(root_path, INPUT, function + "_NT.P"), "rb"
        ) as load_file:
            NT = pickle.load(load_file)
        with open(
            os.path.join(root_path, INPUT, function + "_DT.P"), "rb"
        ) as load_file:
            DT = pickle.load(load_file)

        test_output = read_waveform.calculate_timesteps(NT, DT)

        with open(
            os.path.join(root_path, OUTPUT, function + "_ret_val.P"), "rb"
        ) as load_file:
            bench_output = pickle.load(load_file)

        assert np.isclose(test_output, bench_output).all()


def test_read_waveforms(set_up):
    function = "read_waveforms"
    for root_path in set_up:
        station_names, units = get_common_bbseis_values(root_path, function)
        bbseis, comp, wave_type, file_type = get_common_waveform_values(
            root_path, function
        )

        # only test for binary, path to ascii folder is not needed
        test_output = read_waveform.read_waveforms(
            None, bbseis, station_names, comp, wave_type, file_type, units
        )

        with open(
            os.path.join(root_path, OUTPUT, function + "_ret_val.P"), "rb"
        ) as load_file:
            bench_output = pickle.load(load_file)
        compare_iterable(test_output, bench_output)


def test_read_one_station_from_bbseis(set_up):  # station name not the same
    function = "read_one_station_from_bbseries"
    for root_path in set_up:
        with open(
            os.path.join(root_path, INPUT, function + "_station_name.P"), "rb"
        ) as load_file:
            station_name = pickle.load(load_file)

        bbseis, comp, wave_type, file_type = get_common_waveform_values(
            root_path, function
        )

        test_output = read_waveform.read_one_station_from_bbseries(
            bbseis, station_name, comp, wave_type, file_type
        )

        with open(
            os.path.join(root_path, OUTPUT, function + "_waveform.P"), "rb"
        ) as load_file:
            bench_output = pickle.load(load_file)

        compare_waveforms(bench_output, test_output)


def test_read_binary_file(set_up):
    function = "read_binary_file"
    for root_path in set_up:
        station_names, units = get_common_bbseis_values(root_path, function)
        bbseis, comp, wave_type, file_type = get_common_waveform_values(
            root_path, function
        )

        test_output = read_waveform.read_binary_file(
            bbseis, comp, station_names, wave_type, file_type, units
        )

        with open(
            os.path.join(root_path, OUTPUT, function + "_waveforms.P"), "rb"
        ) as load_file:
            bench_output = pickle.load(load_file)

        for i in range(len(bench_output)):
            for j in range(2):
                compare_waveforms(bench_output[i][j], test_output[i][j])
