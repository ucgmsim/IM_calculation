import argparse
import csv
import io
import os
import pickle

import numpy as np
import pytest
from qcore import utils

import calculate_ims
from test.test_common_set_up import test_data_save_dirs, INPUT, OUTPUT, set_up, compare_dicts

PARSER = argparse.ArgumentParser()
BSC_PERIOD = [0.05, 0.1,  5.0, 10.0]
TEST_IMS = ['PGA', 'PGV', 'Ds575', 'pSA']

FAKE_DIR = 'fake_dir' # should be in set_up module and remove in tear_down module
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
        for root_path in test_data_save_dirs:

            with open(os.path.join(root_path, INPUT, function + '_comp.P'), 'rb') as load_file:
                comp = pickle.load(load_file)

            actual_converted_comp = calculate_ims.convert_str_comp(comp)

            with open(os.path.join(root_path, OUTPUT, function + '_converted_comp.P'), 'rb') as load_file:
                expected_converted_comp = pickle.load(load_file)

            assert actual_converted_comp == expected_converted_comp

    def test_array_to_dict(self):
        function = 'array_to_dict'
        for root_path in test_data_save_dirs:
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
        for root_path in test_data_save_dirs:
            with open(os.path.join(root_path, INPUT, function + '_value_tuple.P'), 'rb') as load_file:
                value_tuple = pickle.load(load_file)

            actual_result = calculate_ims.compute_measure_single(value_tuple)

            with open(os.path.join(root_path, OUTPUT, function + '_result.P'), 'rb') as load_file:
                expected_result = pickle.load(load_file)

            compare_dicts(actual_result, expected_result)

    def test_get_bbseis(self):
        function = 'get_bbseis'
        for root_path in test_data_save_dirs:
            with open(os.path.join(root_path, INPUT, function + '_selected_stations.P'), 'rb') as load_file:
                stations = pickle.load(load_file)

            actual_converted_stations = calculate_ims.get_bbseis(os.path.join(root_path, INPUT, 'BB.bin'), 'binary', stations)[1]

            with open(os.path.join(root_path, OUTPUT, function + '_station_names.P'), 'rb') as load_file:
                expected_converted_stations = pickle.load(load_file)

            assert actual_converted_stations == expected_converted_stations

    def test_compute_measures_multiprocess(self):
        function = 'compute_measures_multiprocess'
        for root_path in test_data_save_dirs:
            with open(os.path.join(root_path, INPUT, function + '_input_path.P'), 'rb') as load_file:
                input_path = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_file_type.P'), 'rb') as load_file:
                file_type = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_geom_only.P'), 'rb') as load_file:
                geom_only = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_wave_type.P'), 'rb') as load_file:
                wave_type = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_station_names.P'), 'rb') as load_file:
                station_names = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_ims.P'), 'rb') as load_file:
                ims = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_comp.P'), 'rb') as load_file:
                comp = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_period.P'), 'rb') as load_file:
                period = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_output.P'), 'rb') as load_file:
                output = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_identifier.P'), 'rb') as load_file:
                identifier = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_rupture.P'), 'rb') as load_file:
                rupture = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_run_type.P'), 'rb') as load_file:
                run_type = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_version.P'), 'rb') as load_file:
                version = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_process.P'), 'rb') as load_file:
                process = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_simple_output.P'), 'rb') as load_file:
                simple_output = pickle.load(load_file)

            calculate_ims.compute_measures_multiprocess(input_path, file_type, geom_only, wave_type, station_names, ims,
                                                        comp, period, output, identifier, rupture, run_type, version,
                                                        process, simple_output)

    def test_get_result_filepath(self):
        function = 'get_result_filepath'
        for root_path in test_data_save_dirs:
            with open(os.path.join(root_path, INPUT, function + '_output_folder.P'), 'rb') as load_file:
                output_folder = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_arg_identifier.P'), 'rb') as load_file:
                arg_identifier = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_suffix.P'), 'rb') as load_file:
                suffix = pickle.load(load_file)

            actual_ret_val = calculate_ims.get_result_filepath(output_folder, arg_identifier, suffix)

            with open(os.path.join(root_path, OUTPUT, function + '_ret_val.P'), 'rb') as load_file:
                expected_ret_val = pickle.load(load_file)

            assert actual_ret_val == expected_ret_val

    def test_get_header(self):
        function = 'get_header'
        for root_path in test_data_save_dirs:
            with open(os.path.join(root_path, INPUT, function + '_ims.P'), 'rb') as load_file:
                ims = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_period.P'), 'rb') as load_file:
                period = pickle.load(load_file)

            actual_header = calculate_ims.get_header(ims, period)

            with open(os.path.join(root_path, OUTPUT, function + '_header.P'), 'rb') as load_file:
                expected_header = pickle.load(load_file)

            assert actual_header == expected_header

    def test_get_comp_name_and_list(self):

        function = 'get_comp_name_and_list'
        for root_path in test_data_save_dirs:
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
        for root_path in test_data_save_dirs:
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

    def test_write_result(self):
        function = 'write_result'
        for root_path in test_data_save_dirs:
            with open(os.path.join(root_path, INPUT, function + '_result_dict.P'), 'rb') as load_file:
                result_dict = pickle.load(load_file)
            output_folder = root_path
            with open(os.path.join(root_path, INPUT, function + '_identifier.P'), 'rb') as load_file:
                identifier = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_comp.P'), 'rb') as load_file:
                comp = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_ims.P'), 'rb') as load_file:
                ims = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_period.P'), 'rb') as load_file:
                period = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_geom_only.P'), 'rb') as load_file:
                geom_only = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_simple_output.P'), 'rb') as load_file:
                simple_output = pickle.load(load_file)

            calculate_ims.write_result(result_dict, output_folder, identifier, comp, ims, period, geom_only, simple_output)

    def test_generate_metadata(self):
        function = 'generate_metadata'
        for root_path in test_data_save_dirs:
            output_folder = root_path
            os.makedirs(os.path.join(root_path, 'stations'), exist_ok=True)
            with open(os.path.join(root_path, INPUT, function + '_identifier.P'), 'rb') as load_file:
                identifier = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_rupture.P'), 'rb') as load_file:
                rupture = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_run_type.P'), 'rb') as load_file:
                run_type = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_version.P'), 'rb') as load_file:
                version = pickle.load(load_file)

            calculate_ims.generate_metadata(output_folder, identifier, rupture, run_type, version)

    def test_get_comp_help(self):
        function = 'get_comp_help'
        for root_path in test_data_save_dirs:
            actual_ret_val = calculate_ims.get_comp_help()

            with open(os.path.join(root_path, OUTPUT, function + '_ret_val.P'), 'rb') as load_file:
                expected_ret_val = pickle.load(load_file)

            assert actual_ret_val == expected_ret_val

    def test_get_im_or_period_help(self):
        function = 'get_im_or_period_help'
        for root_path in test_data_save_dirs:
            with open(os.path.join(root_path, INPUT, function + '_default_values.P'), 'rb') as load_file:
                default_values = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_im_or_period.P'), 'rb') as load_file:
                im_or_period = pickle.load(load_file)

            actual_ret_val = calculate_ims.get_im_or_period_help(default_values, im_or_period)

            with open(os.path.join(root_path, OUTPUT, function + '_ret_val.P'), 'rb') as load_file:
                expected_ret_val = pickle.load(load_file)

            assert actual_ret_val == expected_ret_val

    def test_validate_input_path(self):
        function = 'validate_input_path'
        for root_path in test_data_save_dirs:
            with open(os.path.join(root_path, INPUT, function + '_arg_input.P'), 'rb') as load_file:
                arg_input = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_arg_file_type.P'), 'rb') as load_file:
                arg_file_type = pickle.load(load_file)

            calculate_ims.validate_input_path(PARSER, arg_input, arg_file_type)

    def test_validate_comp(self):
        function = 'validate_comp'
        for root_path in test_data_save_dirs:
            with open(os.path.join(root_path, INPUT, function + '_arg_comp.P'), 'rb') as load_file:
                arg_comp = pickle.load(load_file)

            actual_comp, acutal_geom_only = calculate_ims.validate_comp(PARSER, arg_comp)

            with open(os.path.join(root_path, OUTPUT, function + '_comp.P'), 'rb') as load_file:
                expected_comp = pickle.load(load_file)
            with open(os.path.join(root_path, OUTPUT, function + '_geom_only.P'), 'rb') as load_file:
                expected_geom_only = pickle.load(load_file)

            assert actual_comp == expected_comp
            assert acutal_geom_only == expected_geom_only

    def test_validate_im(self):
        function = 'validate_im'
        for root_path in test_data_save_dirs:
            with open(os.path.join(root_path, INPUT, function + '_arg_im.P'), 'rb') as load_file:
                arg_im = pickle.load(load_file)

            actual_im = calculate_ims.validate_im(PARSER, arg_im)

            with open(os.path.join(root_path, OUTPUT, function + '_im.P'), 'rb') as load_file:
                expected_im = pickle.load(load_file)

            assert actual_im == expected_im

    def test_validate_period(self):
        function = 'validate_period'
        for root_path in test_data_save_dirs:
            with open(os.path.join(root_path, INPUT, function + '_arg_period.P'), 'rb') as load_file:
                arg_period = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_arg_extended_period.P'), 'rb') as load_file:
                arg_extended_period = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_im.P'), 'rb') as load_file:
                im = pickle.load(load_file)

            actual_period = calculate_ims.validate_period(PARSER, arg_period, arg_extended_period, im)

            with open(os.path.join(root_path, OUTPUT, function + '_period.P'), 'rb') as load_file:
                expected_period = pickle.load(load_file)

            assert not (actual_period - expected_period).any()

    def test_get_steps(self):
        function = 'get_steps'
        for root_path in test_data_save_dirs:
            with open(os.path.join(root_path, INPUT, function + '_input_path.P'), 'rb') as load_file:
                input_path = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_nps.P'), 'rb') as load_file:
                nps = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_total_stations.P'), 'rb') as load_file:
                total_stations = pickle.load(load_file)

            actual_steps = calculate_ims.get_steps(input_path, nps, total_stations)

            with open(os.path.join(root_path, OUTPUT, function + '_steps.P'), 'rb') as load_file:
                expected_steps = pickle.load(load_file)

            assert actual_steps == expected_steps
