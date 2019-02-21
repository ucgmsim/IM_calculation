import os
import pickle
import pytest
import numpy as np
import sys
import shutil
from qcore import shared
from IM import read_waveform

# from test.test_common_set_up import set_up, TEST_DATA_SAVE_DIRS, INPUT, OUTPUT

INPUT = "input"
OUTPUT = "output"
REALISATIONS = [
        ('PangopangoF29_HYP01-10_S1244', "https://www.dropbox.com/sh/dgpfukqd01zucjv/AAA8iMASZWn5vbr0PdDCgTG3a?dl=0")]
TEST_DATA_SAVE_DIRS = []

# Run this once, but run it for any test/collection of tests that is run in this class
# @pytest.fixture(scope='session', autouse=True)

def set_up():
    print("dsaf")
    for i, (REALISATION, DATA_DOWNLOAD_PATH) in enumerate(REALISATIONS):
        DATA_STORE_PATH = os.path.join(".", "sample"+str(i))

        ZIP_DOWNLOAD_PATH = os.path.join(DATA_STORE_PATH, REALISATION+".zip")
        OUTPUT_DIR_PATH = os.path.join(DATA_STORE_PATH, "input")

        DOWNLOAD_CMD = "wget -O {} {}".format(ZIP_DOWNLOAD_PATH, DATA_DOWNLOAD_PATH)
        UNZIP_CMD = "unzip {} -d {}".format(ZIP_DOWNLOAD_PATH, DATA_STORE_PATH)
        print(DATA_STORE_PATH)
        TEST_DATA_SAVE_DIRS.append(DATA_STORE_PATH)
        if not os.path.isdir(DATA_STORE_PATH):
            os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
            out, err = shared.exe(DOWNLOAD_CMD, debug=False)
            if b"failed" in err:
                os.remove(ZIP_DOWNLOAD_PATH)
                sys.exit("{} failed to download data folder".format(err))
            else:
                print("Successfully downloaded benchmark data folder")

            out, err = shared.exe(UNZIP_CMD, debug=False)
            os.remove(ZIP_DOWNLOAD_PATH)
            if b"error" in err:
                shutil.rmtree(OUTPUT_DIR_PATH)
                sys.exit("{} failed to extract data folder".format(err))
        else:
            print("Benchmark data folder already exits: ", DATA_STORE_PATH)


    # Remove the test data directory
    #for PATH in TEST_DATA_SAVE_DIRS:
        #shutil.rmtree(PATH)


def get_common_waveform_values(root_path, function_name):
    with open(os.path.join(root_path, INPUT, function_name + '_bbseis.P'), 'rb') as load_file:
        bbseis = pickle.load(load_file)
    with open(os.path.join(root_path, INPUT, function_name + '_comp.P'), 'rb') as load_file:
        comp = pickle.load(load_file)
    with open(os.path.join(root_path, INPUT, function_name + '_wave_type.P'), 'rb') as load_file:
        wave_type = pickle.load(load_file)
    with open(os.path.join(root_path, INPUT, function_name + '_file_type.P'), 'rb') as load_file:
        file_type = pickle.load(load_file)

    return bbseis, comp, wave_type, file_type


def get_common_bbseis_values(root_path, function_name):
    with open(os.path.join(root_path, INPUT, function_name + '_station_names.P'), 'rb') as load_file:
        station_names = pickle.load(load_file)

    with open(os.path.join(root_path, INPUT, function_name + '_units.P'), 'rb') as load_file:
        units = pickle.load(load_file)
    return station_names, units


def test_calculate_timesteps():
    function = 'calculate_timesteps'
    for root_path in TEST_DATA_SAVE_DIRS:
        with open(os.path.join(root_path, INPUT, function + '_NT.P'), 'rb') as load_file:
            NT = pickle.load(load_file)
        with open(os.path.join(root_path, INPUT, function + '_DT.P'), 'rb') as load_file:
            DT = pickle.load(load_file)

        test_output = read_waveform.calculate_timesteps(NT, DT)

        with open(os.path.join(root_path, OUTPUT, function + '_ret_val.P'), 'rb') as load_file:
            bench_output = pickle.load(load_file)

        assert (test_output == bench_output).all()


def test_read_waveforms():
    function = 'read_waveforms'
    for root_path in TEST_DATA_SAVE_DIRS:
        station_names, units = get_common_bbseis_values(root_path, function)
        bbseis, comp, wave_type, file_type = get_common_waveform_values(root_path, function)

        # only test for binary, path to ascii folder is not neede
        test_output = read_waveform.read_waveforms(None, bbseis, station_names, comp, wave_type, file_type, units)

        with open(os.path.join(root_path, OUTPUT, function + '_ret_val.P'), 'rb') as load_file:
            bench_output = pickle.load(load_file)
        for i in range(len(bench_output)):
            for j in range(2):
                vars_test = vars(test_output[i][j])
                vars_bench = vars(bench_output[i][j])
                for k in vars_test.keys():
                    if isinstance(vars_test[k], np.ndarray):
                        assert (vars_test[k] == vars_bench[k]).all()
                    else:
                        assert vars_test[k] == vars_bench[k]



def test_read_one_station_from_bbseis(): #station name not the same
    function = 'read_one_station_from_bbseries'
    print(function)
    for root_path in TEST_DATA_SAVE_DIRS:
        print("root_path",root_path)
        with open(os.path.join(root_path, INPUT, function + '_station_name.P'), 'rb') as load_file:
            station_name = pickle.load(load_file)
            print("station_name from pikle",station_name)

        bbseis, comp, wave_type, file_type = get_common_waveform_values(root_path, function)

        test_output = read_waveform.read_one_station_from_bbseries(bbseis, station_name, comp, wave_type, file_type)

        with open(os.path.join(root_path, OUTPUT, function + '_waveform.P'), 'rb') as load_file:
            bench_output = pickle.load(load_file)
        print("test", test_output.station_name)
        print("bench", bench_output.station_name)
        assert test_output == bench_output


def test_read_binary_file():
    function = 'read_binary_file'
    for root_path in TEST_DATA_SAVE_DIRS:
        station_names, units = get_common_bbseis_values(root_path, function)
        bbseis, comp, wave_type, file_type = get_common_waveform_values(root_path, function)

        test_output = read_waveform.read_binary_file(bbseis, comp, station_names, wave_type, file_type, units)

        with open(os.path.join(root_path, OUTPUT, function + '_waveforms.P'), 'rb') as load_file:
            bench_output = pickle.load(load_file)

        for i in range(len(bench_output)):
            for j in range(2):
                vars_test = vars(test_output[i][j])
                vars_bench = vars(bench_output[i][j])
                for k in vars_test.keys():
                    if isinstance(vars_test[k], np.ndarray):
                        assert (vars_test[k] == vars_bench[k]).all()
                    else:
                        assert vars_test[k] == vars_bench[k]


set_up()
test_read_waveforms()