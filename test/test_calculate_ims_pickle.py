import pickle

import pytest
import shutil

import calculate_ims
import os
import getpass
import sys
from qcore import shared


REALISATION = 'PangopangoF29_HYP01-10_S1244'

DATA_DOWNLOAD_PATH = "https://www.dropbox.com/sh/dgpfukqd01zucjv/AAA8iMASZWn5vbr0PdDCgTG3a?dl=0"
DATA_STORE_PATH = os.path.join("/home", getpass.getuser(), REALISATION)
DOWNLOAD_CMD = "wget -O {} {}".format(DATA_STORE_PATH+'.zip', DATA_DOWNLOAD_PATH)
UNZIP_CMD = "unzip {} -d {}".format(os.path.join("/home", getpass.getuser(), REALISATION+'.zip'), os.path.join("/home", getpass.getuser(), REALISATION))

test_data_save_dir = os.path.join("/home", getpass.getuser(), REALISATION)


@pytest.fixture(scope='session', autouse=True)
def set_up():
    if not os.path.isfile(DATA_STORE_PATH):
        out, err = shared.exe(DOWNLOAD_CMD, debug=False)
        if b"failed" in err:
            os.remove(DATA_STORE_PATH+'.zip')
            sys.exit("{} failed to download data folder".format(err))
        else:
            print("Successfully downloaded benchmark data folder")

        out, err = shared.exe(UNZIP_CMD, debug=False)
        os.remove(DATA_STORE_PATH+'.zip')
        if b"error" in err:
            os.remove(DATA_STORE_PATH + '.zip')
            shutil.rmtree(DATA_STORE_PATH)
            sys.exit("{} failed to extract data folder".format(err))
    else:
        print("Benchmark data folder already exits")


@pytest.fixture(scope='session', autouse=True)
def tear_down():
    yield
    shutil.rmtree(DATA_STORE_PATH)


def test_convert_str_comp():
    function = 'convert_str_comp'
    with open(os.path.join(test_data_save_dir, function + '_comp.P'), 'rb') as load_file:
        comp = pickle.load(load_file)
    value_to_test = calculate_ims.convert_str_comp(comp)
    with open(os.path.join(test_data_save_dir, function + '_converted_comp.P'), 'rb') as load_file:
        converted_comp = pickle.load(load_file)
    assert value_to_test == converted_comp

