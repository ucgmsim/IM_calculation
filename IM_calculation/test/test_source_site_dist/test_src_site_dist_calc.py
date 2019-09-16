import os
import pickle

import numpy as np
import pytest

from qcore import geo

from IM_calculation.source_site_dist.src_site_dist import calc_rrup_rjb, calc_rx_ry
from IM_calculation.test.test_common_set_up import INPUT, set_up, OUTPUT


def test_calc_rrub_rjb(set_up):
    function = "calc_rrup_rjb"
    for root_path in set_up:
        srf_points = np.load(
            os.path.join(root_path, INPUT, function + "_srf_points.npy")
        )
        locations = np.load(os.path.join(root_path, INPUT, function + "_locations.npy"))

        out_rrup = np.load(os.path.join(root_path, OUTPUT, function + "_rrup.npy"))
        out_rjb = np.load(os.path.join(root_path, OUTPUT, function + "_rjb.npy"))

        rrup, rjb = calc_rrup_rjb(srf_points, locations)

        assert np.all(np.isclose(out_rrup, rrup))
        assert np.all(np.isclose(out_rjb, rjb))


BASIC_SRF_POINTS = np.asarray([[0, 0, 0], [1, 0, 0], [0, -1, 1], [1, -1, 1]])
BASIC_SRF_HEADER = [{"centre": [0.5, 0], "nstrike": 2, "ndip": 2, "strike": 90.0}]
BASIC_STATIONS = np.asarray(
    [
        [0, 0, 0],
        [0.5, 0, 0],
        [*geo.ll_shift(0, 0, 50, 0)[::-1], 0],
        [*geo.ll_shift(*geo.ll_shift(0, 0, 50, 270), 100, 0)[::-1], 0],
    ]
)
BASIC_RX = np.asarray([0, 0, -50, -100])
BASIC_RY = np.asarray([0, 0, 0, 50])

HOSSACK_SRF_POINTS = np.asarray(
    [
        [176.2493, -38.3301, 0.0431],
        [176.2202, -38.3495, 0.0431],
        [176.2096, -38.3033, 7.886],
        [176.1814, -38.3221, 7.886],
    ]
)
HOSSACK_SRF_HEADER = [
    {"centre": [176.2354, -38.3404], "nstrike": 2, "ndip": 2, "strike": 230.0}
]
HOSSACK_STATIONS = np.asarray(
    [
        [
            176.16718461,
            -38.27736689,
            0,
        ],  # location 9.118km down dip of the top centre point
        [
            176.30243581,
            -38.40219607,
            0,
        ],  # location 9.118km up dip of the top centre point
        [
            176.17626422,
            -38.35520564,
            0,
        ],  # Location 3.0383km to the South West of the fault
    ]
)
HOSSACK_RX = np.asarray([9.1180, -9.1180, 1.9987])
HOSSACK_RY = np.asarray([0.0, 0.0, 3.0383])

RELATIVE_TOLERANCE = 0.001  # 1m tolerance


@pytest.mark.parametrize(
    ["srf_points", "srf_header", "station_location", "rx_bench", "ry_bench"],
    [
        (BASIC_SRF_POINTS, BASIC_SRF_HEADER, BASIC_STATIONS, BASIC_RX, BASIC_RY),
        (
            HOSSACK_SRF_POINTS,
            HOSSACK_SRF_HEADER,
            HOSSACK_STATIONS,
            HOSSACK_RX,
            HOSSACK_RY,
        ),
    ],
)
def test_calc_rx_ry_basic(srf_points, srf_header, station_location, rx_bench, ry_bench):
    rx, ry = calc_rx_ry(srf_points, srf_header, station_location)
    assert np.all(np.isclose(rx, rx_bench, rtol=RELATIVE_TOLERANCE))
    assert np.all(np.isclose(ry, ry_bench, rtol=RELATIVE_TOLERANCE))


def test_calc_rx_ry(set_up):
    function = "calc_rx_ry"
    for root_path in set_up:
        srf_points = np.load(
            os.path.join(root_path, INPUT, function + "_srf_points.npy")
        )
        srf_header = pickle.load(
            open(os.path.join(root_path, INPUT, function + "_srf_header.P"), "rb")
        )
        locations = np.load(os.path.join(root_path, INPUT, function + "_locations.npy"))

        out_rx = np.load(os.path.join(root_path, OUTPUT, function + "_rx.npy"))
        out_ry = np.load(os.path.join(root_path, OUTPUT, function + "_ry.npy"))

        rx, ry = calc_rx_ry(srf_points, srf_header, locations)

        assert np.all(np.isclose(out_rx, rx))
        assert np.all(np.isclose(out_ry, ry))
