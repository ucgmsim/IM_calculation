import numba
import numpy as np
import pandas as pd

from qcore.geo import get_distances
from qcore.constants import SourceToSiteDist


@numba.jit(parallel=True)
def calc_rrup_rjb(srf_points: np.ndarray, locations: np.ndarray):
    """Calculates rrub and rjb distance

    Parameters
    ----------
    srf_points: np.ndarray
        The fault points from the srf file (qcore, srf.py, read_srf_points),
        format (lon, lat, depth)
    locations: np.ndarray
        The locations for which to calculate the distances,
        format (lon, lat, depth)

    Returns
    -------
    rrups : np.ndarray
        The rrup distance for the locations, shape/order same as locations
    rjb : np.ndarray
        The rjb distance for the locations, shape/order same as locations
    """
    rrups = np.empty(locations.shape[0])
    rjb = np.empty(locations.shape[0])

    h_dist_f = numba.njit(get_distances)
    for loc_ix in numba.prange(locations.shape[0]):
        h_dist = h_dist_f(srf_points, locations[loc_ix, 0], locations[loc_ix, 1])

        v_dist = srf_points[:, -1] - locations[loc_ix, -1]

        d = np.sqrt(h_dist ** 2 + v_dist ** 2)

        rrups[loc_ix] = np.min(d)
        rjb[loc_ix] = np.min(h_dist)

    return rrups, rjb


def write_source_2_site_dists(
    out_file: str,
    stations: np.ndarray,
    locations: np.ndarray,
    r_rup: np.ndarray,
    r_jb: np.ndarray,
    r_x: np.ndarray = None,
):
    """Writes the source to site distances to a csv file"""
    data = [locations[:, 0], locations[:, 1], r_rup, r_jb]
    cols_names = [
        "lon",
        "lat",
        SourceToSiteDist.R_rup.str_value,
        SourceToSiteDist.R_jb.str_value,
    ]

    if r_x is not None:
        data.append(r_x)
        cols_names.append(SourceToSiteDist.R_x.str_value)

    data = np.asarray(data).T

    df = pd.DataFrame(data=data, columns=cols_names, index=stations)
    df.to_csv(out_file, index_label="station")