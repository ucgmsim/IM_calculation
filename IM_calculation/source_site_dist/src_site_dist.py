from typing import List, Dict

import numba
import numpy as np

from qcore.geo import get_distances, ll_cross_track_dist


@numba.jit(parallel=True)
def calc_rrup_rjb(srf_points: np.ndarray, locations: np.ndarray):
    """Calculates rrup and rjb distance

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


@numba.jit(parallel=True)
def calc_rx(srf_points: np.ndarray, plane_infos: List[Dict], locations: np.ndarray):
    """Calculates r_x distance using the cross track distance calculation"""
    r_x = np.empty(locations.shape[0])
    h_dist_f = numba.njit(get_distances)
    for iloc in range(locations.shape[0]):
        lon, lat, _ = locations[iloc]

        # Get the closest point on the fault plane, currently assumes Location of interest
        h_dist = h_dist_f(srf_points, lon, lat)
        d = np.sqrt(h_dist ** 2 + srf_points[:, -1] ** 2)

        # Get the index of closest fault plane point
        point_ix = np.argmin(d)
        p_lon, p_lat, _ = srf_points[point_ix]

        # Have to work out which plane the point belongs to
        # Get cumulative number of points in each plane
        n_points_cumulative = np.asarray(
            [plane["nstrike"] * plane["ndip"] for plane in plane_infos]
        ).cumsum()
        # Check which planes have points with index greater than the nearest point
        greater_than_threshold = n_points_cumulative > point_ix
        # Get the first of these planes
        plane_ix = greater_than_threshold.searchsorted(True)

        r_x[iloc] = ll_cross_track_dist(
            p_lon,
            p_lat,
            None,
            None,
            lon,
            lat,
            a12=np.deg2rad(plane_infos[plane_ix]["strike"]),
        )

    return r_x
