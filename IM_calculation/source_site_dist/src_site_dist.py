from typing import List, Dict

import numba
import numpy as np

from qcore.geo import get_distances, ll_cross_track_dist, ll_bearing, radians, R_EARTH, ll_dist
import itertools

#@numba.jit(parallel=True)
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


def calc_cart_cross_track(x1, y1, x2, y2, x0, y0):
    x1, y1, x2, y2, x0, y0 = np.abs((x1, y1, x2, y2, x0, y0)) - np.abs((x1, y1, x1, y1, x1, y1))
    print((x1, y1, x2, y2, x0, y0))
    area = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    dist = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    print(area, dist)
    return radians(area / dist)*R_EARTH


#@numba.jit(parallel=True)
def calc_rx_ry(srf_points: np.ndarray, plane_infos: List[Dict], locations: np.ndarray):
    """Calculates r_x distance using the cross track distance calculation"""
    r_x = np.empty(locations.shape[0])
    r_y = np.empty(locations.shape[0])
    h_dist_f = numba.njit(get_distances)

    extended_points = np.r_[srf_points, [[0, 0, 0]]]

    # Seperate the srf points into the different planes
    pnt_counts = [plane["nstrike"]*plane["ndip"] for plane in plane_infos]
    pnt_counts.insert(0, 0)
    pnt_counts = np.cumsum(pnt_counts)
    pnt_sections = [extended_points[pnt_counts[i]:pnt_counts[i+1]] for i in range(len(plane_infos))]

    # Get the top/bottom edges of each plane
    top_edges = [section[section[:, 2] == min(section[:, 2])] for section in pnt_sections]
    bottom_edges = [section[section[:, 2] == max(section[:, 2])] for section in pnt_sections]

    for iloc in range(locations.shape[0]):
        lon, lat, _ = locations[iloc]

        # Get the closest point on the fault plane
        h_dist = h_dist_f(srf_points, lon, lat)
        # print(h_dist)

        # Get the index of closest fault plane point
        point_ix = np.argmin(h_dist)

        # Have to work out which plane the point belongs to
        # Get cumulative number of points in each plane

        # Check which planes have points with index greater than the nearest point
        greater_than_threshold = pnt_counts > point_ix
        # print(greater_than_threshold)

        # Get the first of these planes
        plane_ix = greater_than_threshold.searchsorted(True)-1

        # print(top_edges[offset_n_points_cumulative[plane_ix], :2], top_edges[offset_n_points_cumulative[plane_ix+1]-1, :2])

        up_strike_top_point = top_edges[plane_ix][0, :2]
        up_strike_bottom_point = bottom_edges[plane_ix][0, :2]
        down_strike_top_point = top_edges[plane_ix][-1, :2]
        down_strike_bottom_point = bottom_edges[plane_ix][-1, :2]

        # If the angle from the first point to the second point is not within 10 degrees of the strike,
        # then we should swap the two points
        if not np.isclose(ll_bearing(*up_strike_top_point, *down_strike_top_point), plane_infos[plane_ix]["strike"], atol=10):
            print("Swapping")
            up_strike_top_point, down_strike_top_point = down_strike_top_point, up_strike_top_point
            up_strike_bottom_point, down_strike_bottom_point = down_strike_bottom_point, up_strike_bottom_point

        r_x[iloc] = ll_cross_track_dist(
            *up_strike_top_point,
            *down_strike_top_point,
            lon,
            lat,
        )

        print("Up strike", up_strike_top_point, up_strike_bottom_point)
        up_strike_dist = abs(ll_cross_track_dist(
            *up_strike_top_point,
            *up_strike_bottom_point,
            lon,
            lat,
        ))
        print("Down strike: ", down_strike_top_point, down_strike_bottom_point)
        down_strike_dist = abs(ll_cross_track_dist(
            *down_strike_top_point,
            *down_strike_bottom_point,
            lon,
            lat,
        ))

        if np.isclose(up_strike_dist+down_strike_dist, plane_infos[plane_ix]["width"]):
            r_y[iloc] = 0
        else:
            print("Ry")
            print(up_strike_dist, down_strike_dist)
            print(*up_strike_top_point,
            *up_strike_bottom_point,)
            print(*down_strike_top_point,
            *down_strike_bottom_point,)
            r_y[iloc] = min(up_strike_dist, down_strike_dist)
            print("Ry done")

        print("Cart cross track: ", plane_ix, calc_cart_cross_track(
            *up_strike_top_point,
            *down_strike_top_point,
            lon,
            lat,
        ))

        for pair in itertools.combinations([up_strike_top_point, down_strike_top_point, up_strike_bottom_point, down_strike_bottom_point], r=2):
            pt1, pt2 = pair
            pt_bear = ll_bearing(*pt1, *pt2)
            if 270 < pt_bear or 90 > pt_bear:
                pt_bear = (pt_bear + 180) % 360
            print(pt1, pt2, ll_dist(*pt1, *pt2), pt_bear)

    return r_x, r_y


def calc_rx_header(srf_points: np.ndarray, plane_infos: List[Dict], locations: np.ndarray):
    """Calculates r_x distance using the cross track distance calculation"""
    r_x = np.empty(locations.shape[0])
    h_dist_f = numba.njit(get_distances)

    extended_points = np.r_[srf_points, [[0, 0, 0]]]

    # Seperate the srf points into the different planes
    pnt_counts = [plane["nstrike"]*plane["ndip"] for plane in plane_infos]
    pnt_counts.insert(0, 0)
    pnt_counts = np.cumsum(pnt_counts)
    pnt_sections = [extended_points[pnt_counts[i]:pnt_counts[i+1]] for i in range(len(plane_infos))]

    # Get the top edges of each plane
    top_edges = np.concatenate([section[section[:, 2] == min(section[:, 2])] for section in pnt_sections])
    # print(len(top_edges))

    # Get the srf point index ranges
    n_points_cumulative = np.asarray(
        [plane["nstrike"] for plane in plane_infos]
    ).cumsum()
    offset_n_points_cumulative = [0]
    offset_n_points_cumulative.extend(n_points_cumulative)

    for iloc in range(locations.shape[0]):
        lon, lat, _ = locations[iloc]

        # Get the closest point on the fault plane
        h_dist = h_dist_f(srf_points, lon, lat)
        # print(h_dist)

        # Get the index of closest fault plane point
        point_ix = np.argmin(h_dist)

        # Have to work out which plane the point belongs to
        # Get cumulative number of points in each plane

        # Check which planes have points with index greater than the nearest point
        greater_than_threshold = pnt_counts > point_ix
        # print(greater_than_threshold)

        # Get the first of these planes
        plane_ix = greater_than_threshold.searchsorted(True)-1

        r_x[iloc] = ll_cross_track_dist(
            *plane_infos[plane_ix]["centre"],
            None,
            None,  # *second_point,
            lon,
            lat,
            a12=plane_infos[plane_ix]["strike"]
        )

        clon, clat = plane_infos[plane_ix]["centre"]
        pclon = clon+np.sin(radians(plane_infos[plane_ix]["strike"]))
        pclat = clat+np.cos(radians(plane_infos[plane_ix]["strike"]))
        print("Points: ", clon, clat, pclon, pclat)
        print("Header cart cross track: ", plane_ix, calc_cart_cross_track(
            clon,
            clat,
            pclon,
            pclat,
            lon,
            lat,
        ))

    return r_x
