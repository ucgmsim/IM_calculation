import os
import sys
import numpy as np
import argparse
import pytest
sys.path.insert(0, '../../')
import calculate_ims


PARSER = argparse.ArgumentParser()
BSC_PERIOD = [0.05, 0.1,  5.0, 10.0]
TEST_IMS = ['PGA', 'PGV', 'Ds575', 'pSA']


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