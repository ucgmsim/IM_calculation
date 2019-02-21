import numpy as np
import argparse
import pickle
import pytest
import os
import io
import csv

import calculate_ims
from IM import read_waveform
from qcore import utils

from test.test_common_set_up import TEST_DATA_SAVE_DIRS, INPUT, OUTPUT

PARSER = argparse.ArgumentParser()
BSC_PERIOD = [0.05, 0.1,  5.0, 10.0]
TEST_IMS = ['PGA', 'PGV', 'Ds575', 'pSA']

FAKE_DIR = 'fake_dir'  # should be created in set_up module and remove in tear_down module
utils.setup_dir("fake_dir")


@pytest.mark.parametrize(
    "test_comp, expected_comp", [('ellipsis', 'ellipsis'), ('000', '000'), ('090', '090'), ('ver', 'ver'), ('geom', 'ellipsis')]
)
def test_validate_comp(test_comp, expected_comp):
    assert calculate_ims.validate_comp(PARSER, test_comp)[0] == expected_comp


@pytest.mark.parametrize("test_comp_fail", ["adsf"])
def test_validate_comp_fail(test_comp_fail):
    with pytest.raises(SystemExit):
         calculate_ims.validate_comp(PARSER, test_comp_fail)


@pytest.mark.parametrize(
    "test_period, test_extended, test_im, expected_period", [(BSC_PERIOD, False, TEST_IMS, np.array(BSC_PERIOD)), (BSC_PERIOD, True, TEST_IMS, np.unique(np.append(BSC_PERIOD, calculate_ims.EXT_PERIOD)))]
)
def test_validate_period(test_period, test_extended, test_im, expected_period):
    assert all(np.equal(calculate_ims.validate_period(PARSER, test_period, test_extended, test_im), expected_period))


@pytest.mark.parametrize(
    "test_period, test_extended, test_im", [(BSC_PERIOD, False, TEST_IMS[:-1]), (BSC_PERIOD, True, TEST_IMS[:-1])]
)
def test_validate_period_fail(test_period, test_extended, test_im):
    with pytest.raises(SystemExit):
        calculate_ims.validate_period(PARSER, test_period, test_extended, test_im)


@pytest.mark.parametrize(
    "test_path, test_file_type", [("asdf", 'b'), (FAKE_DIR, 'b')]
)
def test_validate_input_path_fail(test_path, test_file_type):
    with pytest.raises(SystemExit):
        calculate_ims.validate_input_path(PARSER, test_path, test_file_type)


class TestPickleTesting():
    def test_convert_str_comp(self):

        function = 'convert_str_comp'
        for root_path in TEST_DATA_SAVE_DIRS:

            with open(os.path.join(root_path, INPUT, function + '_comp.P'), 'rb') as load_file:
                comp = pickle.load(load_file)

            actual_converted_comp = calculate_ims.convert_str_comp(comp)

            with open(os.path.join(root_path, OUTPUT, function + '_converted_comp.P'), 'rb') as load_file:
                expected_converted_comp = pickle.load(load_file)

            assert actual_converted_comp == expected_converted_comp

    def test_get_comp_name_and_list(self):

        function = 'get_comp_name_and_list'
        for root_path in TEST_DATA_SAVE_DIRS:
            with open(os.path.join(root_path, INPUT, function + '_comp.P'), 'rb') as load_file:
                comp = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_geom_only.P'), 'rb') as load_file:
                geom_only = pickle.load(load_file)

            actual_comp_name, actual_comps = calculate_ims.get_comp_name_and_list(comp, geom_only)

            with open(os.path.join(root_path, OUTPUT, function + '_comp_name.P'), 'rb') as load_file:
                expected_comp_name = pickle.load(load_file)
            with open(os.path.join(root_path, OUTPUT, function + '_comps.P'), 'rb') as load_file:
                expected_comps = pickle.load(load_file)

            assert actual_comp_name == expected_comp_name
            assert actual_comps == expected_comps

    def test_write_rows(self):

        function = 'write_rows'
        for root_path in TEST_DATA_SAVE_DIRS:
            with open(os.path.join(root_path, INPUT, function + '_comps.P'), 'rb') as load_file:
                comps = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_station.P'), 'rb') as load_file:
                station = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_ims.P'), 'rb') as load_file:
                ims = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_result_dict.P'), 'rb') as load_file:
                result_dict = pickle.load(load_file)

            big_csv = io.StringIO()
            big_csv_writer = csv.writer(big_csv)
            sub_csv_writer = csv.writer(io.StringIO())

            calculate_ims.write_rows(comps, station, ims, result_dict, big_csv_writer, sub_csv_writer=sub_csv_writer)

    def test_get_bbseis(self):
        function = 'get_bbseis'
        for root_path in TEST_DATA_SAVE_DIRS:
            with open(os.path.join(root_path, INPUT, function + '_selected_stations.P'), 'rb') as load_file:
                stations = pickle.load(load_file)

            actual_converted_stations = calculate_ims.get_bbseis(os.path.join(root_path, INPUT, 'BB.bin'), 'binary', stations)[1]

            with open(os.path.join(root_path, OUTPUT, function + '_station_names.P'), 'rb') as load_file:
                expected_converted_stations = pickle.load(load_file)

            assert actual_converted_stations == expected_converted_stations

    def test_array_to_dict(self):
        function = 'array_to_dict'
        for root_path in TEST_DATA_SAVE_DIRS:
            with open(os.path.join(root_path, INPUT, function + '_value.P'), 'rb') as load_file:
                value = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_comp.P'), 'rb') as load_file:
                comp = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_converted_comp.P'), 'rb') as load_file:
                converted_comp = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_im.P'), 'rb') as load_file:
                im = pickle.load(load_file)

            actual_value_dict = calculate_ims.array_to_dict(value, comp, converted_comp, im)

            with open(os.path.join(root_path, OUTPUT, function + '_value_dict.P'), 'rb') as load_file:
                expected_value_dict = pickle.load(load_file)

            assert actual_value_dict == expected_value_dict

    def test_compute_measure_single(self):
        function = 'compute_measure_single'
        for root_path in TEST_DATA_SAVE_DIRS:
            with open(os.path.join(root_path, INPUT, function + '_value_tuple.P'), 'rb') as load_file:
                value_tuple = pickle.load(load_file)

            actual_result = calculate_ims.compute_measure_single(value_tuple)

            with open(os.path.join(root_path, OUTPUT, function + '_result.P'), 'rb') as load_file:
                expected_result = pickle.load(load_file)

            assert actual_result == expected_result

    def test_compute_measures_multiprocess(self):
        function = 'compute_measures_multiprocess'
        for root_path in TEST_DATA_SAVE_DIRS:
            with open(os.path.join(root_path, INPUT, function + '_value_tuple.P'), 'rb') as load_file:
                value_tuple = pickle.load(load_file)

            actual_result = calculate_ims.compute_measure_single(value_tuple)

            with open(os.path.join(root_path, OUTPUT, function + '_result.P'), 'rb') as load_file:
                expected_result = pickle.load(load_file)

            #assert actual_result == expected_result

    # read_waveforms
    def get_common_waveform_values(self, root_path, function_name):

        with open(os.path.join(root_path, INPUT, function_name + '_bbseis.P'), 'rb') as load_file:
            bbseis = pickle.load(load_file)
        with open(os.path.join(root_path, INPUT, function_name + '_comp.P'), 'rb') as load_file:
            comp = pickle.load(load_file)
        with open(os.path.join(root_path, INPUT, function_name + '_wave_type.P'), 'rb') as load_file:
            wave_type = pickle.load(load_file)
        with open(os.path.join(root_path, INPUT, function_name + '_file_type.P'), 'rb') as load_file:
            file_type = pickle.load(load_file)

        return bbseis, comp, wave_type, file_type

    def get_common_bbseis_values(self, root_path, function_name):

        with open(os.path.join(root_path, INPUT, function_name + '_station_names.P'), 'rb') as load_file:
            station_names = pickle.load(load_file)

        with open(os.path.join(root_path, INPUT, function_name + '_units.P'), 'rb') as load_file:
            units = pickle.load(load_file)
        return station_names, units

    def test_read_waveforms(self):
        function = 'read_waveforms'
        for root_path in TEST_DATA_SAVE_DIRS:
            with open(os.path.join(root_path, INPUT, function + '_path.P'), 'rb') as load_file:
                path = pickle.load(load_file)
            station_names, units = self.get_common_bbseis_values(root_path,function)
            bbseis, comp, wave_type, file_type = self.get_common_waveform_values(root_path, function)

            test_ouput = read_waveform.read_waveforms(path, bbseis, station_names, comp, wave_type, file_type, units)

            with open(os.path.join(root_path, INPUT, function + '_ret_val.P'), 'rb') as load_file:
                bench_output = pickle.load(load_file)

            assert test_ouput == bench_output

    def test_read_one_station_from_bbseis(self):
        function = 'read_one_station_from_bbseries'
        for root_path in TEST_DATA_SAVE_DIRS:

            with open(os.path.join(root_path, INPUT, function + '_station_name.P'), 'rb') as load_file:
                station_name = pickle.load(load_file)

            bbseis, comp, wave_type, file_type = self.get_common_waveform_values(root_path, function)

            test_output = read_waveform.read_one_station_from_bbseries(bbseis, station_name, comp, wave_type, file_type)

            with open(os.path.join(root_path, INPUT, function + '_waveform.P'), 'rb') as load_file:
                bench_output = pickle.load(load_file)

            assert test_output == bench_output

    def test_read_binary_file(self):
        function = 'read_binary_file'
        for root_path in TEST_DATA_SAVE_DIRS:
            station_names, units = self.get_common_bbseis_values(root_path, function)
            bbseis, comp, wave_type, file_type = self.get_common_waveform_values(root_path, function)

            test_output = read_waveform.read_binary_file(bbseis, comp, station_names, wave_type, file_type, units)

            with open(os.path.join(root_path, INPUT, function + '_waveforms.P'), 'rb') as load_file:
                bench_output = pickle.load(load_file)

            assert test_output == bench_output
