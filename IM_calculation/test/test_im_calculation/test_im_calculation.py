import argparse
import filecmp
import os
import pickle

import pytest
import numpy as np
from qcore.constants import Components

import IM_calculation.IM.im_calculation as calculate_ims
from qcore import utils, constants
from IM_calculation.test.test_common_set_up import INPUT, OUTPUT, compare_dicts, set_up

# This is a hack, to allow loading of the test pickle objects
import sys
import IM_calculation.IM as IM

sys.modules["IM"] = IM


PARSER = argparse.ArgumentParser()
BSC_PERIOD = [0.05, 0.1, 5.0, 10.0]
TEST_IMS = ["PGA", "PGV", "Ds575", "pSA"]

FAKE_DIR = (
    "fake_dir"  # should be created in set_up module and remove in tear_down module
)
utils.setup_dir("fake_dir")


@pytest.mark.parametrize(
    "test_period, test_extended, expected_period",
    [
        (BSC_PERIOD, False, np.array(BSC_PERIOD)),
        (BSC_PERIOD, True, np.unique(np.append(BSC_PERIOD, constants.EXT_PERIOD))),
    ],
)
def test_validate_period(test_period, test_extended, expected_period):
    assert all(
        np.equal(
            calculate_ims.validate_period(test_period, test_extended), expected_period
        )
    )


@pytest.mark.parametrize("test_path, test_file_type", [("asdf", "b"), (FAKE_DIR, "b")])
@pytest.mark.mpi_skip
def test_validate_input_path_fail(test_path, test_file_type):
    with pytest.raises(SystemExit):
        calculate_ims.validate_input_path(PARSER, test_path, test_file_type)


def convert_str_comps_to_enum(expected_result):
    for station in expected_result.keys():
        for im in expected_result[station].keys():
            if im == "pSA":
                for comp in list(expected_result[station][im][1]):
                    expected_result[station][im][1][
                        Components.from_str(comp)
                    ] = expected_result[station][im][1][comp]
                    del expected_result[station][im][1][comp]
            else:
                for comp in list(expected_result[station][im]):
                    expected_result[station][im][
                        Components.from_str(comp)
                    ] = expected_result[station][im][comp]
                    del expected_result[station][im][comp]


class TestPickleTesting:
    @pytest.mark.mpi_skip
    def test_convert_str_comp(self, set_up):
        function = "convert_str_comp"
        for root_path in set_up:

            with open(
                os.path.join(root_path, INPUT, function + "_comp.P"), "rb"
            ) as load_file:
                comp = pickle.load(load_file)

            int_comp, str_comp = Components.get_comps_to_calc_and_store(comp)

            with open(
                os.path.join(root_path, OUTPUT, function + "_str_comp_for_int.P"), "rb"
            ) as load_file:
                expected_int_comp = pickle.load(load_file)
            with open(
                os.path.join(root_path, OUTPUT, function + "_str_comp.P"), "rb"
            ) as load_file:
                expected_str_comp = pickle.load(load_file)

            assert [x.str_value for x in int_comp] == expected_int_comp
            assert [x.str_value for x in str_comp] == expected_str_comp

    @pytest.mark.mpi_skip
    def test_array_to_dict(self, set_up):
        function = "array_to_dict"
        for root_path in set_up:
            with open(
                os.path.join(root_path, INPUT, function + "_value.P"), "rb"
            ) as load_file:
                value = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_comp.P"), "rb"
            ) as load_file:
                arg_comps = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_str_comp.P"), "rb"
            ) as load_file:
                str_comps = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_im.P"), "rb"
            ) as load_file:
                im = pickle.load(load_file)

            str_comps = [Components.from_str(x) for x in str_comps]
            arg_comps = [Components.from_str(x) for x in arg_comps]
            actual_value_dict = calculate_ims.array_to_dict(
                value, str_comps, im, arg_comps
            )

            with open(
                os.path.join(root_path, OUTPUT, function + "_value_dict.P"), "rb"
            ) as load_file:
                expected_value_dict = pickle.load(load_file)

            assert actual_value_dict == expected_value_dict

    @pytest.mark.mpi_skip
    def test_compute_measure_single(self, set_up):
        function = "compute_measure_single"
        for root_path in set_up:
            with open(
                os.path.join(root_path, INPUT, function + "_value_tuple.P"), "rb"
            ) as load_file:
                value_tuple = pickle.load(load_file)
            waveform, ims, comps, periods, str_comps = value_tuple
            im_options = {"pSA": periods}
            comps = [Components.from_str(x) for x in comps]
            str_comps = [Components.from_str(x) for x in str_comps]
            actual_result = calculate_ims.compute_measure_single(
                waveform, ims, comps, im_options, str_comps, (0, 0)
            )

            with open(
                os.path.join(root_path, OUTPUT, function + "_result.P"), "rb"
            ) as load_file:
                expected_result = pickle.load(load_file)
                convert_str_comps_to_enum(expected_result)
                actual_expected_result = self.convert_to_results_dict(
                    periods, expected_result
                )

            compare_dicts(actual_result, actual_expected_result)

    @pytest.mark.mpi_skip
    def test_get_bbseis(self, set_up):
        function = "get_bbseis"
        for root_path in set_up:
            with open(
                os.path.join(root_path, INPUT, function + "_selected_stations.P"), "rb"
            ) as load_file:
                stations = pickle.load(load_file)

            actual_converted_stations = calculate_ims.get_bbseis(
                os.path.join(root_path, INPUT, "BB.bin"), "binary", stations
            )[1]

            with open(
                os.path.join(root_path, OUTPUT, function + "_station_names.P"), "rb"
            ) as load_file:
                expected_converted_stations = pickle.load(load_file)

            assert actual_converted_stations == expected_converted_stations

    @pytest.mark.mpi(min_size=2)
    def test_compute_measures_multiprocess(self, set_up):
        from mpi4py import MPI

        function = "compute_measures_multiprocess"
        for root_path in set_up:
            input_path = os.path.join(root_path, INPUT, "BB.bin")
            with open(
                os.path.join(root_path, INPUT, function + "_file_type.P"), "rb"
            ) as load_file:
                file_type = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_wave_type.P"), "rb"
            ) as load_file:
                wave_type = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_ims.P"), "rb"
            ) as load_file:
                ims = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_comp.P"), "rb"
            ) as load_file:
                comp = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_period.P"), "rb"
            ) as load_file:
                period = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_identifier.P"), "rb"
            ) as load_file:
                identifier = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_rupture.P"), "rb"
            ) as load_file:
                rupture = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_run_type.P"), "rb"
            ) as load_file:
                run_type = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_version.P"), "rb"
            ) as load_file:
                version = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_simple_output.P"), "rb"
            ) as load_file:
                simple_output = pickle.load(load_file)
            station_names = ["099A"]
            output = root_path
            os.makedirs(os.path.join(output, "stations"), exist_ok=True)

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            calculate_ims.compute_measures_mpi(
                input_path,
                file_type,
                comm,
                rank,
                size,
                wave_type=wave_type,
                station_names=station_names,
                ims=ims,
                comp=comp,
                im_options={"pSA": period},
                output=output,
                identifier=identifier,
                rupture=rupture,
                run_type=run_type,
                version=version,
                simple_output=simple_output,
            )

    @pytest.mark.mpi_skip
    def test_get_result_filepath(self, set_up):
        function = "get_result_filepath"
        for root_path in set_up:
            with open(
                os.path.join(root_path, INPUT, function + "_output_folder.P"), "rb"
            ) as load_file:
                output_folder = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_arg_identifier.P"), "rb"
            ) as load_file:
                arg_identifier = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_suffix.P"), "rb"
            ) as load_file:
                suffix = pickle.load(load_file)

            actual_ret_val = calculate_ims.get_result_filepath(
                output_folder, arg_identifier, suffix
            )

            with open(
                os.path.join(root_path, OUTPUT, function + "_ret_val.P"), "rb"
            ) as load_file:
                expected_ret_val = pickle.load(load_file)

            assert actual_ret_val == expected_ret_val

    @pytest.mark.mpi_skip
    def test_write_result(self, set_up):
        function = "write_result"
        for root_path in set_up:
            with open(
                os.path.join(root_path, INPUT, function + "_period.P"), "rb"
            ) as load_file:
                period = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_result_dict.P"), "rb"
            ) as load_file:
                temp_result_dict = pickle.load(load_file)
                convert_str_comps_to_enum(temp_result_dict)
                result_dict = self.convert_to_results_dict(
                    period, temp_result_dict, keep_ps=True
                )

            with open(
                os.path.join(root_path, INPUT, function + "_identifier.P"), "rb"
            ) as load_file:
                identifier = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_simple_output.P"), "rb"
            ) as load_file:
                simple_output = pickle.load(load_file)

            output_folder = root_path

            os.makedirs(
                os.path.join(output_folder, calculate_ims.OUTPUT_SUBFOLDER),
                exist_ok=True,
            )

            calculate_ims.write_result(
                result_dict, output_folder, identifier, simple_output
            )
            expected_output_path = calculate_ims.get_result_filepath(
                output_folder, identifier, ".csv"
            )
            actual_output_path = os.path.join(
                root_path, OUTPUT, function + "_outfile.csv"
            )
            expected_output = np.loadtxt(
                expected_output_path, delimiter=",", usecols=range(2, 24), skiprows=1
            )
            actual_output = np.loadtxt(
                actual_output_path, delimiter=",", usecols=range(2, 24), skiprows=1
            )

            assert np.isclose(expected_output, actual_output).all()

    @pytest.mark.mpi_skip
    def convert_to_results_dict(self, period, temp_result_dict, keep_ps=False):
        result_dict = {}
        for station in sorted(temp_result_dict):
            temp_result_dict[station]["pSA"] = temp_result_dict[station]["pSA"][1]
            for im in sorted(temp_result_dict[station]):
                for comp in temp_result_dict[station][im]:
                    if (station, comp.str_value) not in result_dict:
                        result_dict[(station, comp.str_value)] = {}
                    if im in calculate_ims.MULTI_VALUE_IMS:
                        for i, val in enumerate(period):
                            if keep_ps:
                                result_dict[(station, comp.str_value)][
                                    f"{im}_{str(val).replace('.', 'p')}"
                                ] = temp_result_dict[station][im][comp][i]
                            else:

                                result_dict[(station, comp.str_value)][
                                    f"{im}_{str(val)}"
                                ] = temp_result_dict[station][im][comp][i]
                    else:
                        result_dict[(station, comp.str_value)][im] = temp_result_dict[
                            station
                        ][im][comp]
        return result_dict

    @pytest.mark.mpi_skip
    def test_generate_metadata(self, set_up):
        function = "generate_metadata"
        for root_path in set_up:

            with open(
                os.path.join(root_path, INPUT, function + "_identifier.P"), "rb"
            ) as load_file:
                identifier = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_rupture.P"), "rb"
            ) as load_file:
                rupture = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_run_type.P"), "rb"
            ) as load_file:
                run_type = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_version.P"), "rb"
            ) as load_file:
                version = pickle.load(load_file)

            # Save to the realisations folder that will be deleted after the run has finished
            output_folder = root_path

            calculate_ims.generate_metadata(
                output_folder, identifier, rupture, run_type, version
            )

            actual_output_path = calculate_ims.get_result_filepath(
                output_folder, identifier, "_imcalc.info"
            )
            expected_output_path = os.path.join(
                root_path, OUTPUT, function + "_outfile.info"
            )

            filecmp.cmp(actual_output_path, expected_output_path)

    @pytest.mark.mpi_skip
    def test_validate_input_path(self, set_up):
        function = "validate_input_path"
        for root_path in set_up:
            arg_input = os.path.join(root_path, INPUT, "BB.bin")
            with open(
                os.path.join(root_path, INPUT, function + "_arg_file_type.P"), "rb"
            ) as load_file:
                arg_file_type = pickle.load(load_file)

            calculate_ims.validate_input_path(PARSER, arg_input, arg_file_type)
            # Function does not return anything, only raises errors through the parser

    @pytest.mark.mpi_skip
    def test_validate_period(self, set_up):
        function = "validate_period"
        for root_path in set_up:
            with open(
                os.path.join(root_path, INPUT, function + "_arg_period.P"), "rb"
            ) as load_file:
                arg_period = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_arg_extended_period.P"),
                "rb",
            ) as load_file:
                arg_extended_period = pickle.load(load_file)

            actual_period = calculate_ims.validate_period(
                arg_period, arg_extended_period
            )

            with open(
                os.path.join(root_path, OUTPUT, function + "_period.P"), "rb"
            ) as load_file:
                expected_period = pickle.load(load_file)

            assert (actual_period == expected_period).all()

    @pytest.mark.mpi_skip
    def test_get_steps(self, set_up):
        function = "get_steps"
        for root_path in set_up:
            input_path = os.path.join(root_path, INPUT, "BB.bin")
            with open(
                os.path.join(root_path, INPUT, function + "_nps.P"), "rb"
            ) as load_file:
                nps = pickle.load(load_file)
            with open(
                os.path.join(root_path, INPUT, function + "_total_stations.P"), "rb"
            ) as load_file:
                total_stations = pickle.load(load_file)

            actual_steps = calculate_ims.get_steps(input_path, nps, total_stations)

            with open(
                os.path.join(root_path, OUTPUT, function + "_steps.P"), "rb"
            ) as load_file:
                expected_steps = pickle.load(load_file)

            assert actual_steps == expected_steps
