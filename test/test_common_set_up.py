import os
from qcore import shared
import shutil
import sys
import pytest
import numpy as np

from rrup.rrup import Point
from IM.read_waveform import Waveform

INPUT = "input"
OUTPUT = "output"
REALISATIONS = [
    (
        "PangopangoF29_HYP01-10_S1244",
        "https://www.dropbox.com/sh/dgpfukqd01zucjv/AAA8iMASZWn5vbr0PdDCgTG3a?dl=0",
    )
]


# Run this once, but run it for any test/collection of tests that is run in this class
@pytest.yield_fixture(scope="module", autouse=True)
def set_up(request):
    test_data_save_dirs = []
    for i, (REALISATION, DATA_DOWNLOAD_PATH) in enumerate(REALISATIONS):
        data_store_path = os.path.join(request.fspath.dirname, "sample" + str(i))

        zip_download_path = os.path.join(data_store_path, REALISATION + ".zip")

        download_cmd = "wget -O {} {}".format(zip_download_path, DATA_DOWNLOAD_PATH)
        unzip_cmd = "unzip {} -d {}".format(zip_download_path, data_store_path)
        # print(DATA_STORE_PATH)
        test_data_save_dirs.append(data_store_path)
        if not os.path.isdir(data_store_path):
            os.makedirs(data_store_path, exist_ok=True)
            out, err = shared.exe(download_cmd, debug=False)
            if b"failed" in err:
                os.remove(zip_download_path)
                sys.exit("{} failed to download data folder".format(err))
            else:
                print("Successfully downloaded benchmark data folder")

            out, err = shared.exe(unzip_cmd, debug=False)
            os.remove(zip_download_path)
            if b"error" in err:
                shutil.rmtree(data_store_path)
                sys.exit("{} failed to extract data folder".format(err))

        else:
            print("Benchmark data folder already exits: ", data_store_path)

    # Run all tests
    yield test_data_save_dirs

    # Remove the test data directory
    for PATH in test_data_save_dirs:
        shutil.rmtree(PATH)


def compare_waveforms(bench_waveform, test_waveform):
    assert isinstance(bench_waveform, Waveform)
    assert isinstance(test_waveform, Waveform)
    vars_test = vars(test_waveform)
    vars_bench = vars(bench_waveform)
    for k in vars_bench.keys():
        if isinstance(vars_bench[k], np.ndarray):
            assert np.isclose(vars_test[k], vars_bench[k]).all()
        else:
            assert vars_test[k] == vars_bench[k]


def compare_points(actual_point, expected_point):
    assert isinstance(actual_point, Point)
    assert isinstance(expected_point, Point)
    assert actual_point.Lat == expected_point.Lat
    assert actual_point.Lon == expected_point.Lon
    assert actual_point.Depth == expected_point.Depth
    assert actual_point.r_rups == expected_point.r_rups
    assert actual_point.r_jbs == expected_point.r_jbs


def compare_dicts(actual_result, expected_result):

    assert isinstance(actual_result, dict)
    assert isinstance(expected_result, dict)
    assert actual_result.keys() == expected_result.keys()

    for key in actual_result.keys():
        if isinstance(actual_result[key], dict) or isinstance(
            expected_result[key], dict
        ):
            compare_dicts(actual_result[key], expected_result[key])
        elif isinstance(actual_result[key], (list, tuple)) or isinstance(
            expected_result[key], (list, tuple)
        ):
            compare_iterable(actual_result[key], expected_result[key])
        elif isinstance(actual_result[key], np.ndarray) or isinstance(
            expected_result[key], np.ndarray
        ):
            assert not (actual_result[key] - expected_result[key]).any()
        elif isinstance(actual_result[key], Point) or isinstance(
            expected_result[key], Point
        ):
            compare_points(actual_result[key], expected_result[key])
        elif isinstance(actual_result[key], Waveform) or isinstance(
            expected_result[key], Waveform
        ):
            compare_waveforms(actual_result[key], expected_result[key])
        else:
            assert actual_result[key] == expected_result[key]


def compare_iterable(actual_result, expected_result):
    assert len(actual_result) == len(expected_result)
    assert type(actual_result) == type(expected_result)

    for i in range(len(actual_result)):
        if isinstance(actual_result[i], dict) or isinstance(expected_result[i], dict):
            compare_dicts(actual_result[i], expected_result[i])
        elif isinstance(actual_result[i], (list, tuple)) or isinstance(
            expected_result[i], (list, tuple)
        ):
            compare_iterable(actual_result[i], expected_result[i])
        elif isinstance(actual_result[i], np.ndarray) or isinstance(
            expected_result[i], np.ndarray
        ):
            assert not (actual_result[i] - expected_result[i]).any()
        elif isinstance(actual_result[i], Point) or isinstance(
            expected_result[i], Point
        ):
            compare_points(actual_result[i], expected_result[i])
        elif isinstance(actual_result[i], Waveform) or isinstance(
            expected_result[i], Waveform
        ):
            compare_waveforms(actual_result[i], expected_result[i])
        else:
            assert actual_result[i] == expected_result[i]
