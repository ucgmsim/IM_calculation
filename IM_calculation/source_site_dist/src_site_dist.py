from typing import List, Dict

import matplotlib.path as mpltPath
import numba
import numpy as np

from qcore.geo import get_distances, ll_cross_track_dist, ll_bearing

numba.config.THREADING_LAYER = "omp"
h_dist_f = numba.njit(get_distances)

VOLCANIC_FRONT_COORDS = [(175.508, -39.364), (177.199, -37.73)]
VOLCANIC_FRONT_LINE = mpltPath.Path(VOLCANIC_FRONT_COORDS)


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

    for loc_ix in numba.prange(locations.shape[0]):
        h_dist = h_dist_f(srf_points, locations[loc_ix, 0], locations[loc_ix, 1])

        v_dist = srf_points[:, 2] - locations[loc_ix, 2]

        d = np.sqrt(h_dist ** 2 + v_dist ** 2)

        rrups[loc_ix] = np.min(d)
        rjb[loc_ix] = np.min(h_dist)

    return rrups, rjb


def calc_rx_ry(srf_points: np.ndarray, plane_infos: List[Dict], locations: np.ndarray, system="GC1"):
    if system == "GC1":
        return calc_rx_ry_GC1(srf_points, plane_infos, locations)
    elif system == "GC2":
        return calc_rx_ry_GC2(srf_points, plane_infos, locations[..., :2])
    else:
        raise ValueError("Illegal system value")


def calc_rx_ry_GC1(srf_points: np.ndarray, plane_infos: List[Dict], locations: np.ndarray):
    """Calculates r_x distance using the cross track distance calculation"""
    r_x = np.empty(locations.shape[0])
    r_y = np.empty(locations.shape[0])

    extended_points = np.r_[srf_points, [[0, 0, 0]]]

    # Separate the srf points into the different planes
    pnt_counts = [plane["nstrike"] * plane["ndip"] for plane in plane_infos]
    pnt_counts.insert(0, 0)
    pnt_counts = np.cumsum(pnt_counts)
    pnt_sections = [
        extended_points[pnt_counts[i] : pnt_counts[i + 1]]
        for i in range(len(plane_infos))
    ]

    # Get the top/bottom edges of each plane
    top_edges = [
        section[: header["nstrike"]]
        for section, header in zip(pnt_sections, plane_infos)
    ]
    bottom_edges = [
        section[-header["nstrike"] :]
        for section, header in zip(pnt_sections, plane_infos)
    ]

    for iloc in range(locations.shape[0]):
        lon, lat, *_ = locations[iloc]

        if len(plane_infos) > 1:
            # Have to work out which plane the point belongs to
            # Get cumulative number of points in each plane

            # Get the closest point in the fault
            h_dist = h_dist_f(srf_points, lon, lat)

            # Get the index of closest fault plane point
            point_ix = np.argmin(h_dist)

            # Check which planes have points with index greater than the nearest point
            greater_than_threshold = pnt_counts > point_ix

            # Get the first of these planes
            plane_ix = greater_than_threshold.searchsorted(True) - 1

        else:
            # If there is only one plane we don't need to find the nearest plane
            plane_ix = 0

        up_strike_top_point = top_edges[plane_ix][0, :2]
        up_strike_bottom_point = bottom_edges[plane_ix][0, :2]
        down_strike_top_point = top_edges[plane_ix][-1, :2]
        down_strike_bottom_point = bottom_edges[plane_ix][-1, :2]

        # If the angle from the first point to the second point is not within 10 degrees of the strike,
        # then we should swap the two points
        if not np.isclose(
            ll_bearing(*up_strike_top_point, *down_strike_top_point),
            plane_infos[plane_ix]["strike"],
            atol=10,
        ):
            up_strike_top_point, down_strike_top_point = (
                down_strike_top_point,
                up_strike_top_point,
            )
            up_strike_bottom_point, down_strike_bottom_point = (
                down_strike_bottom_point,
                up_strike_bottom_point,
            )

        r_x[iloc] = ll_cross_track_dist(
            *up_strike_top_point, *down_strike_top_point, lon, lat
        )

        up_strike_dist = ll_cross_track_dist(
            *up_strike_top_point, *up_strike_bottom_point, lon, lat
        )
        down_strike_dist = ll_cross_track_dist(
            *down_strike_top_point, *down_strike_bottom_point, lon, lat
        )

        if np.sign(up_strike_dist) != np.sign(down_strike_dist):
            # If the signs are different then the point is between the lines projected along the edges of the srf
            r_y[iloc] = 0
        else:
            r_y[iloc] = np.min(np.abs([up_strike_dist, down_strike_dist]))

    return r_x, r_y


def calc_backarc(srf_points: np.ndarray, locations: np.ndarray):
    """
    This is a crude approximation of stations that are on the backarc. Defined by source-site lines that cross the
    Volcanic front line.
    https://user-images.githubusercontent.com/25143301/111406807-ce5bb600-8737-11eb-9c78-b909efe7d9db.png
    https://user-images.githubusercontent.com/25143301/111408728-93a74d00-873a-11eb-9afa-5e8371ee2504.png

    srf_points: np.ndarray
        The fault points from the srf file (qcore, srf.py, read_srf_points),
        format (lon, lat, depth)
    locations: np.ndarray
        The locations for which to calculate the distances,
        format (lon, lat, depth)
    :return: a numpy array returning 0 if the station is on the forearc and 1 if the station is on the backarc
    """
    n_locations = locations.shape[0]
    backarc = np.zeros(n_locations, dtype=np.int)
    for loc_index in range(n_locations):
        # Selection is every 40 SRF points (4 km) - the backarc line is ~200km long.
        # In the case of point sources it will just take the first point
        for srf_point in srf_points[::40]:
            srf_stat_line = mpltPath.Path(
                [
                    (srf_point[0], srf_point[1]),
                    (locations[loc_index][0], locations[loc_index][1]),
                ]
            )
            if VOLCANIC_FRONT_LINE.intersects_path(srf_stat_line):
                backarc[loc_index] = 1
                break
    return backarc


def calc_rx_ry_GC2(srf_points: np.ndarray, plane_infos: List[Dict], locations: np.ndarray, use_first=False):
    """
    Calculates the rx and ry values for a given fault and list of locations
    :param srf_points: A (n, 3) array giving the lon, lat, depth of each of the n points in the fault srf
    :param plane_infos: A list of plane header dictionaries, one dictionary for each plane in the fault
    :param locations: A (m, 2) array giving the lon, lat of each of m stations/locations to find the rx/ry values for
    :return:
    """

    # Local import to prevent openquake being a full dependency of any importing script
    try:
        from openquake.hazardlib.geo.surface.simple_fault import SimpleFaultSurface
        from openquake.hazardlib.geo.surface.multi import MultiSurface
        from openquake.hazardlib.geo.mesh import Mesh, RectangularMesh
    except ImportError as e:
        print(
            "openquake.engine does not seem to be installed. Use 'pip install openquake.engine' to install it"
        )
        raise e

    # print(plane_infos)

    # Separate the srf points into the different planes
    pnt_counts = [plane["nstrike"] * plane["ndip"] for plane in plane_infos]
    pnt_sections = np.split(srf_points, pnt_counts[:-1])
    pnt_sections = [
        plane.reshape((plane_header["nstrike"], plane_header["ndip"], 3))
        for plane, plane_header in zip(pnt_sections, plane_infos)
    ]

    planes = []
    for plane in pnt_sections:
        fault_points = np.split(plane, 3, axis=-1)
        squeezed_locs = [loc.squeeze() for loc in fault_points]
        planes.append(SimpleFaultSurface(RectangularMesh(*squeezed_locs)))
        # for x in squeezed_locs:
        #     print(np.unique(x))
    # print("Planes len: ", len(planes))
    fault_object = MultiSurface(planes)

    split_locations = [x.squeeze() for x in np.split(locations, 2, axis=-1)]

    location_mesh = Mesh(*split_locations)

    if use_first:
        r_x = planes[0].get_rx_distance(location_mesh)
        r_y = planes[0].get_ry0_distance(location_mesh)
    else:
        r_x = fault_object.get_rx_distance(location_mesh)
        r_y = fault_object.get_ry0_distance(location_mesh)

    # print("Diff rx: ", r_x-r_x1)
    # print("Diff ry: ", r_y-r_y1)
    # print("Edges ", fault_object.cartesian_edges)

    return r_x, r_y
