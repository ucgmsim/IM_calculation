import os
from qcore import shared
import shutil
import sys
import pytest
import numpy as np

INPUT = "input"
OUTPUT = "output"
REALISATIONS = [
        ('PangopangoF29_HYP01-10_S1244', "https://www.dropbox.com/sh/dgpfukqd01zucjv/AAA8iMASZWn5vbr0PdDCgTG3a?dl=0")]
test_data_save_dirs = []

# Run this once, but run it for any test/collection of tests that is run in this class
@pytest.fixture(scope='session', autouse=True)
def set_up():
    for i, (REALISATION, DATA_DOWNLOAD_PATH) in enumerate(REALISATIONS):
        DATA_STORE_PATH = os.path.join(".", "sample"+str(i))

        ZIP_DOWNLOAD_PATH = os.path.join(DATA_STORE_PATH, REALISATION+".zip")
        OUTPUT_DIR_PATH = os.path.join(DATA_STORE_PATH, "input")

        DOWNLOAD_CMD = "wget -O {} {}".format(ZIP_DOWNLOAD_PATH, DATA_DOWNLOAD_PATH)
        UNZIP_CMD = "unzip {} -d {}".format(ZIP_DOWNLOAD_PATH, DATA_STORE_PATH)
        print(DATA_STORE_PATH)
        test_data_save_dirs.append(DATA_STORE_PATH)
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

    # Run all tests
    yield

    # Remove the test data directory
    #for PATH in test_data_save_dirs:
        #shutil.rmtree(PATH)


def compare_dicts(actual_result, expected_result):

    assert isinstance(actual_result, dict)
    assert isinstance(expected_result, dict)
    assert actual_result.keys() == expected_result.keys()

    for key in actual_result.keys():
        if isinstance(actual_result[key], dict) or isinstance(expected_result[key], dict):
            compare_dicts(actual_result[key], expected_result[key])
        elif isinstance(actual_result[key], (list, tuple)) or isinstance(expected_result[key], (list, tuple)):
            compare_iterable(actual_result[key], expected_result[key])
        elif isinstance(actual_result[key], np.ndarray) or isinstance(expected_result[key], np.ndarray):
            assert not (actual_result[key] - expected_result[key]).any()
        else:
            assert actual_result[key] == expected_result[key]


def compare_iterable(actual_result, expected_result):
    assert len(actual_result) == len(expected_result)

    for i in range(len(actual_result)):
        if isinstance(actual_result[i], dict) or isinstance(expected_result[i], dict):
            compare_dicts(actual_result[i], expected_result[i])
        elif isinstance(actual_result[i], (list, tuple)) or isinstance(expected_result[i], (list, tuple)):
            compare_iterable(actual_result[i], expected_result[i])
        elif isinstance(actual_result[i], np.ndarray) or isinstance(expected_result[i], np.ndarray):
            assert not (actual_result[i] - expected_result[i]).any()
        else:
            assert actual_result[i] == expected_result[i]