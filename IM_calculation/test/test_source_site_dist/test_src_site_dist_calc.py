import os
import numpy as np

from IM_calculation.source_site_dist.src_site_dist import calc_rrup_rjb
from IM_calculation.test.test_common_set_up import INPUT, set_up, OUTPUT


def test_calc_rrub_rjb(set_up):
    function = "calc_rrup_rjb"
    for root_path in set_up:
        srf_points = np.load(os.path.join(root_path, INPUT, function + "_srf_points.npy"))
        locations = np.load(os.path.join(root_path, INPUT, function + "_locations.npy"))

        out_rrup = np.load(os.path.join(root_path, OUTPUT, function + "_rrup.npy"))
        out_rjb = np.load(os.path.join(root_path, OUTPUT, function + "_rjb.npy"))

        rrup, rjb = calc_rrup_rjb(srf_points, locations)

        assert np.all(np.isclose(out_rrup, rrup))
        assert np.all(np.isclose(out_rjb, rjb))






