import pytest
from datetime import datetime
import os
import numpy as np
import sys
import getpass
import shutil
import errno
from qcore import srf, shared

TEST_FOLDER = os.path.abspath(os.path.dirname(__file__))
SCRIPT = os.path.abspath(os.path.join(TEST_FOLDER, '..', '..', 'calculate_ims.py'))
print(SCRIPT)
INPUT_DIR = os.path.join(TEST_FOLDER,'sample1','input')
OUTPUT_DIR = os.path.join(TEST_FOLDER,'sample1','output')
IDENTIFIER = 'darfield_obs'
COMP = 'geom'
STATIONS = ['ASHS', 'BFZ', 'CCC', 'CHHC', 'CMHS', 'DCZ', 'HORC', 'LINC', '']
# def setup_module(scope="module"):
#     """ create a tmp directory for storing output from test"""
#     print "----------setup_module----------"
#     try:
#         os.mkdir(OUTPUT_DIR)
#     except OSError as e:
#         if e.errno != errno.EEXIST:
#             raise


def test_calculate_ims():
    """ test calculate_ims.py """
    print "---------test_calculate_ims------------"
    print("python {} {} a -o {} -i {} -c {}".format(SCRIPT, INPUT_DIR, OUTPUT_DIR, IDENTIFIER, COMP))
    out, err = shared.exe("python {} {} a -o {} -i {} -c {}".format(SCRIPT, INPUT_DIR, OUTPUT_DIR, IDENTIFIER, COMP))  # the most important function to execute the whole script
    assert err == ''

#
# def teardown_module():
#     """ delete the tmp directory if it is empty"""
#     print "---------teardown_module------------"
#     if len(os.listdir(OUTPUT_DIR)) == 0:
#         try:
#             shutil.rmtree(OUTPUT_DIR)
#         except (IOError, OSError) as (e):
#             sys.exit(e)


def test_im_values(station_name):

