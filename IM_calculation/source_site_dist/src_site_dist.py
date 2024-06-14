from typing import List, Dict

import matplotlib.path as mpltPath
import numpy as np

from qcore import geo

VOLCANIC_FRONT_COORDS = [(175.508, -39.364), (177.199, -37.73)]
VOLCANIC_FRONT_LINE = mpltPath.Path(VOLCANIC_FRONT_COORDS)


def calc_rrup_rjb(
    srf_points: np.ndarray,
    locations: np.ndarray,
    n_stations_per_iter: int = 1000,
    return_rrup_points: bool = False,
):
    """Calculates rrup and rjb distance

    Parameters
    ----------
    srf_points: np.ndarray
        The fault points from the srf file (qcore, srf.py, read_srf_points),
        format (lon, lat, depth)
    locations: np.ndarray
        The locations for which to calculate the distances,
        format (lon, lat, depth)
    n_stations_per_iter: int
        Number of stations to iterate over, default to 1000.
        Change based on memory requirements
    return_rrup_points: bool (optional) default False
        If True, returns the lon, lat, depth of the rrup points on the srf

    Returns
    -------
    rrups : np.ndarray
        The rrup distance for the locations, shape/order same as locations
    rjb : np.ndarray
        The rjb distance for the locations, shape/order same as locations
    rrups_points : np.ndarray (optional)
        The lon, lat, depth of the rrup points, shape/order same as locations
    """
    rrups = np.empty(locations.shape[0], dtype=np.float32)
    rjb = np.empty(locations.shape[0], dtype=np.float32)
    rrup_points = np.empty((locations.shape[0], 3), dtype=np.float32)

    srf_points = srf_points.astype(np.float32)
    locations = locations.astype(np.float32)

    # Split locations by a given limit for memory capacity
    split_locations = np.split(
        locations,
        np.arange(n_stations_per_iter, locations.shape[0], n_stations_per_iter),
    )

    for ix, cur_locations in enumerate(split_locations):
        h_dist = np.atleast_2d(
            geo.get_distances(srf_points, cur_locations[:, 0], cur_locations[:, 1])
        )

        v_dist = srf_points[:, 2, np.newaxis] - cur_locations[:, 2]

        d = np.sqrt(h_dist**2 + v_dist.T**2)

        start_ix = n_stations_per_iter * ix
        end_ix = start_ix + cur_locations.shape[0]
        rrups[start_ix:end_ix] = np.min(d, axis=1)
        rjb[start_ix:end_ix] = np.min(h_dist, axis=1)

        if return_rrup_points:
            rrup_points[start_ix:end_ix] = srf_points[np.argmin(d, axis=1)]

    if return_rrup_points:
        return rrups, rjb, rrup_points
    else:
        return rrups, rjb


def calc_rx_ry(
    srf_points: np.ndarray,
    plane_infos: List[Dict],
    locations: np.ndarray,
    hypocentre_origin=False,
    type=2,
):
    """
    A wrapper script allowing external function calls to resolve to the correct location
    :param srf_points: An array with shape (n, 3) giving the lon, lat, depth location of each subfault
    :param plane_infos: A list of srf header dictionaries, as retrieved from qcore.srf.get_headers with idx=True
    :param locations: An array with shape (m, 2) giving the lon, lat locations of each location to get Rx, Ry values for
    :param type: Allows switching between the two GC types if desired
    :param hypocentre_origin: If True sets the Ry origin/0 point to the fault trace projection of the hypocentre. If
    false the most upstrike subfault of the first fault trace is used. Only used for GC2.
    :return: An array with shape (m, 2) giving the Rx, Ry values for each of the given locations
    """
    if type == 1:
        return calc_rx_ry_GC1(srf_points, plane_infos, locations)
    elif type == 2:
        return calc_rx_ry_GC2(
            srf_points, plane_infos, locations, hypocentre_origin=hypocentre_origin
        )
    else:
        raise ValueError(f"Invalid GC type. {type} not in {{1,2}}")


def calc_rx_ry_GC1(
    srf_points: np.ndarray, plane_infos: List[Dict], locations: np.ndarray
):
    """
    Calculates Rx and Ry distances using the cross track and along track distance calculations
    Uses the plane nearest to each of the given locations if there are multiple
    :param srf_points: An array with shape (n, 3) giving the lon, lat, depth location of each subfault
    :param plane_infos: A list of srf header dictionaries, as retrieved from qcore.srf.get_headers with idx=True
    :param locations: An array with shape (m, 2) giving the lon, lat locations of each location to get Rx, Ry values for
    :return: An array with shape (m, 2) giving the Rx, Ry values for each of the given locations
    """
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

    for iloc in range(locations.shape[0]):
        lon, lat, *_ = locations[iloc]

        if len(plane_infos) > 1:
            # Have to work out which plane the point belongs to
            # Get cumulative number of points in each plane

            # Get the closest point in the fault
            h_dist = geo.get_distances(srf_points, lon, lat)

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
        down_strike_top_point = top_edges[plane_ix][-1, :2]

        # If the angle from the first point to the second point is not within 10 degrees of the strike,
        # then we should swap the two points
        if not np.isclose(
            geo.ll_bearing(*up_strike_top_point, *down_strike_top_point),
            plane_infos[plane_ix]["strike"],
            atol=10,
        ):
            up_strike_top_point, down_strike_top_point = (
                down_strike_top_point,
                up_strike_top_point,
            )

        r_x[iloc], r_y[iloc] = geo.ll_cross_along_track_dist(
            *up_strike_top_point, *down_strike_top_point, lon, lat
        )

    return r_x, r_y


def calc_rx_ry_GC2(
    srf_points: np.ndarray,
    plane_infos: List[Dict],
    locations: np.ndarray,
    hypocentre_origin=False,
):
    """
    Calculates Rx and Ry distances using the cross track and along track distance calculations
    If there are multiple fault planes the Rx, Ry values are calculated for each fault plane individually, then weighted
    according to plane length and distance to the location
    For one fault plane this is the same as the GC1 function
    Wrapper that calls calc_rx_ry_GC2_multi_hypocentre
    :param srf_points: An array with shape (n, 3) giving the lon, lat, depth location of each subfault
    :param plane_infos: A list of srf header dictionaries, as retrieved from qcore.srf.get_headers with idx=True
    :param locations: An array with shape (m, 2) giving the lon, lat locations of each location to get Rx, Ry values for
    :param hypocentre_origin: If True sets the Ry origin/0 point to the fault trace projection of the hypocentre. If
    false the most upstrike subfault of the first fault trace is used
    :return: An array with shape (m, 2) giving the Rx, Ry values for each of the given locations
    """
    origin_offset = 0
    # Changes the offset from the centre if hypocentre_origin is specified
    # Is extracted from the plane_infos
    if hypocentre_origin:
        length = sum([plane["length"] for plane in plane_infos])
        # Our faults only use one hypocentre
        # Will only use the first one found if there are multiple
        for plane in plane_infos:
            origin_offset += plane["length"]
            if plane["shyp"] != -999.9000:
                origin_offset += plane["shyp"] - plane["length"] / 2
                break
        origin_offset -= length / 2
    r_x, r_y = calc_rx_ry_GC2_multi_hypocentre(
        srf_points, plane_infos, locations, origin_offsets=np.asarray([origin_offset])
    )
    return r_x[0], r_y[0]


def calc_rx_ry_GC2_multi_hypocentre(
    srf_points: np.ndarray,
    plane_infos: List[Dict],
    locations: np.ndarray,
    origin_offsets: np.ndarray = np.asarray([0]),
):
    """
    Vectorised version of the GC2 calculation along multiple hypocentre locations.
    Calculates Rx and Ry distances using the cross track and along track distance calculations
    If there are multiple fault planes the Rx, Ry values are calculated for each fault plane individually, then weighted
    according to plane length and distance to the location
    For one fault plane this is the same as the GC1 function
    :param srf_points: An array with shape (n, 3) giving the lon, lat, depth location of each subfault
    :param plane_infos: A list of srf header dictionaries, as retrieved from qcore.srf.get_headers with idx=True
    :param locations: An array with shape (m, 2) giving the lon, lat locations of each location to get Rx, Ry values for
    :param origin_offsets: An array with shape (o) with the along strike hypocentre locations from the centre of
    the fault for multiple realisations. If not set then only one origin will be considered at the centre position.
    :return: Two arrays with shape (o, m) giving the Rx, Ry values for each of the given locations for each hypocentre
    """
    # Separate the srf points into the different plane traces
    pnt_counts = [plane["nstrike"] * plane["ndip"] for plane in plane_infos]
    pnt_counts.insert(0, 0)
    pnt_counts = np.cumsum(pnt_counts)
    pnt_sections = [
        srf_points[pnt_counts[i] : pnt_counts[i] + header["nstrike"]]
        for i, header in enumerate(plane_infos)
    ]

    # Adjust the offsets
    length = sum([plane["length"] for plane in plane_infos])
    origin_offsets = -(length / 2 + origin_offsets)

    offsets = origin_offsets
    weights = np.zeros(len(locations))
    r_x_values = np.zeros((len(offsets), len(locations)))
    r_y_values = np.zeros((len(offsets), len(locations)))

    for plane_points, plane_header in zip(pnt_sections, plane_infos):
        r_x_p, r_y_p = calc_rx_ry_GC1(plane_points, [plane_header], locations)
        dists = np.atleast_2d(
            geo.get_distances(plane_points, locations[:, 0], locations[:, 1])
        )
        # Minimum distance of 0.001km to prevent nans/infs
        # A bit hacky, but it works. Only needed when a location is directly on top of a subfault
        dists = np.maximum(dists, 0.001)
        weights_p = np.sum(np.power(dists, -2), axis=1)

        weights += weights_p
        r_x_values[:] += weights_p * r_x_p
        # Reshape to allow broadcasting of the offsets to create the (offsets, locations) shape matrix
        r_y_values[:] += weights_p * (
            r_y_p.reshape(1, len(locations)) + offsets.reshape(len(offsets), 1)
        )
        offsets += plane_header["length"]

    r_x = r_x_values / weights
    r_y = r_y_values / weights

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
