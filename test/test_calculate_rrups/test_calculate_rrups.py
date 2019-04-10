import os
import pickle
import argparse

import calculate_rrups
from test.test_common_set_up import INPUT, OUTPUT, set_up


def test_get_fd_stations(set_up):
    func_name = 'get_fd_stations'
    for root_path in set_up:
        test_output = calculate_rrups.get_fd_stations(os.path.join(root_path, INPUT, "sample_station.ll"))
        with open(
                os.path.join(root_path, OUTPUT, func_name + "_stations.P"), "rb"
        ) as load_file:
            bench_output = pickle.load(load_file)

        assert test_output == bench_output


def test_get_match_stations(set_up):
    func_name = 'get_match_stations'
    for root_path in set_up:
        with open(
            os.path.join(root_path, INPUT, func_name + "_arg_fd.P"), "rb"
        ) as load_file:
            arg_fd = pickle.load(load_file)
        with open(
            os.path.join(root_path, INPUT, func_name + "_arg_stations.P"), "rb"
        ) as load_file:
            arg_stations = pickle.load(load_file)

        test_output = calculate_rrups.get_match_stations(argparse.ArgumentParser(), arg_fd, arg_stations)

        with open(
            os.path.join(root_path, OUTPUT, func_name + "_match_stations.P"), "rb"
        ) as load_file:
            bench_output = pickle.load(load_file)

        assert test_output == bench_output

