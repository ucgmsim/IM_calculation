import os
import pickle
import pytest
import numpy as np

from IM import read_waveform, intensity_measures
from test.test_common_set_up import TEST_DATA_SAVE_DIRS, INPUT, OUTPUT, set_up, compare_dicts


def get_common_spectral_vals(root_path, function_name):
    with open(os.path.join(root_path, INPUT, function_name + '_acceleration.P'), 'rb') as load_file:
        acc = pickle.load(load_file)

    with open(os.path.join(root_path, INPUT, function_name + '_period.P'), 'rb') as load_file:
        period = pickle.load(load_file)

    with open(os.path.join(root_path, INPUT, function_name + '_NT.P'), 'rb') as load_file:
        NT = pickle.load(load_file)

    with open(os.path.join(root_path, INPUT, function_name + '_DT.P'), 'rb') as load_file:
        DT = pickle.load(load_file)

    return acc, period, NT, DT


def get_common_vals(root_path, function_name):
    with open(os.path.join(root_path, INPUT, function_name + '_acceleration.P'), 'rb') as load_file:
        acc = pickle.load(load_file)

    with open(os.path.join(root_path, INPUT, function_name + '_times.P'), 'rb') as load_file:
        times = pickle.load(load_file)

    return acc, times


def get_common_ds_vals(root_path, function_name):
    with open(os.path.join(root_path, INPUT, function_name + '_dt.P'), 'rb') as load_file:
        dt = pickle.load(load_file)

    with open(os.path.join(root_path, INPUT, function_name + '_percLow.P'), 'rb') as load_file:
        perc_low = pickle.load(load_file)

    with open(os.path.join(root_path, INPUT, function_name + '_percHigh.P'), 'rb') as load_file:
        perc_high = pickle.load(load_file)

    return dt, perc_low, perc_high


# fail test[4.448219  2.0940204 1.7136909]
#      bench [5.9166474 8.233998  3.1157265]
def test_get_max_nd():
    function = 'get_max_nd'
    for root_path in TEST_DATA_SAVE_DIRS:
        with open(os.path.join(root_path, INPUT, function + '_data.P'), 'rb') as load_file:
            data = pickle.load(load_file)
        test_output = intensity_measures.get_max_nd(data)

        with open(os.path.join(root_path, OUTPUT, function + '_ret_val.P'), 'rb') as load_file:
            bench_output = pickle.load(load_file)

        assert np.isclose(test_output, bench_output).all()


def test_get_spectral_acceleration():
    function = 'get_spectral_acceleration'
    for root_path in TEST_DATA_SAVE_DIRS:
        acc, period, NT, DT = get_common_spectral_vals(root_path, function)
        test_output = intensity_measures.get_spectral_acceleration(acc, period, NT, DT)

        with open(os.path.join(root_path, OUTPUT, function + '_ret_val.P'), 'rb') as load_file:
            bench_output = pickle.load(load_file)

        assert np.isclose(test_output, bench_output).all()


def test_get_spectral_acceleration_nd():
    function = 'get_spectral_acceleration_nd'
    for root_path in TEST_DATA_SAVE_DIRS:
        acc, period, NT, DT = get_common_spectral_vals(root_path, function)
        test_output = intensity_measures.get_spectral_acceleration_nd(acc, period, NT, DT)

        with open(os.path.join(root_path, OUTPUT, function + '_values.P'), 'rb') as load_file:
            bench_output = pickle.load(load_file)

        assert np.isclose(test_output, bench_output).all()


def test_get_cumulative_abs_velocity_nd():
    function = 'get_cumulative_abs_velocity_nd'
    for root_path in TEST_DATA_SAVE_DIRS:
        acc, times = get_common_vals(root_path, function)
        test_output = intensity_measures.get_cumulative_abs_velocity_nd(acc, times)

        with open(os.path.join(root_path, OUTPUT, function + '_ret_val.P'), 'rb') as load_file:
            bench_output = pickle.load(load_file)

        assert np.isclose(test_output, bench_output).all()


def test_get_arias_intensity_nd():
    function = 'get_arias_intensity_nd'
    for root_path in TEST_DATA_SAVE_DIRS:
        acc, times = get_common_vals(root_path, function)
        with open(os.path.join(root_path, INPUT, function + '_g.P'), 'rb') as load_file:
            g = pickle.load(load_file)

        test_output = intensity_measures.get_arias_intensity_nd(acc, g, times)

        with open(os.path.join(root_path, OUTPUT, function + '_ret_val.P'), 'rb') as load_file:
            bench_output = pickle.load(load_file)

        assert np.isclose(test_output, bench_output).all()


def test_calculate_MMI_nd():
    function = 'calculate_MMI_nd'
    for root_path in TEST_DATA_SAVE_DIRS:
        with open(os.path.join(root_path, INPUT, function + '_velocities.P'), 'rb') as load_file:
            vel = pickle.load(load_file)
        test_output = intensity_measures.calculate_MMI_nd(vel)

        with open(os.path.join(root_path, OUTPUT, function + '_ret_val.P'), 'rb') as load_file:
            bench_output = pickle.load(load_file)

        assert np.isclose(test_output, bench_output).all()


def test_getDs():
    function = 'getDs'
    for root_path in TEST_DATA_SAVE_DIRS:
        dt, perc_low, perc_high = get_common_ds_vals(root_path, function)
        with open(os.path.join(root_path, INPUT, function + '_fx.P'), 'rb') as load_file:
            fx = pickle.load(load_file)
        test_output = intensity_measures.getDs(dt, fx, perc_low, perc_high)

        with open(os.path.join(root_path, OUTPUT, function + '_Ds.P'), 'rb') as load_file:
            bench_output = pickle.load(load_file)

        assert np.isclose(test_output, bench_output).all()


def test_getDs_nd():
    function = 'getDs_nd'
    for root_path in TEST_DATA_SAVE_DIRS:
        dt, perc_low, perc_high = get_common_ds_vals(root_path, function)
        with open(os.path.join(root_path, INPUT, function + '_accelerations.P'), 'rb') as load_file:
            acc = pickle.load(load_file)
        test_output = intensity_measures.getDs_nd(dt, acc, perc_low, perc_high)
        with open(os.path.join(root_path, OUTPUT, function + '_values.P'), 'rb') as load_file:
            bench_output = pickle.load(load_file)

        assert np.isclose(test_output, bench_output).all()


def test_get_geom():
    function = 'get_geom'
    for root_path in TEST_DATA_SAVE_DIRS:
        with open(os.path.join(root_path, INPUT, function + '_d1.P'), 'rb') as load_file:
            d1 = pickle.load(load_file)
        with open(os.path.join(root_path, INPUT, function + '_d2.P'), 'rb') as load_file:
            d2 = pickle.load(load_file)

        test_output = intensity_measures.get_geom(d1, d2)

        with open(os.path.join(root_path, OUTPUT, function + '_ret_val.P'), 'rb') as load_file:
            bench_output = pickle.load(load_file)

        assert np.isclose(test_output, bench_output).all()


@pytest.mark.parametrize(
    "test_d1, test_d2, expected_geom", [(0, 0, 0), (1, 5, 2.236067)]
)
def test_get_geom_params(test_d1, test_d2, expected_geom):
    assert np.isclose(intensity_measures.get_geom(test_d1, test_d2), expected_geom).all()


