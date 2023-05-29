#!/usr/bin/env python3
"""Script for calculating rrup and rjb for the specified fault and stations"""
import argparse
import numpy as np
import pandas as pd
from qcore.srf import read_srf_points, read_header
from qcore.formats import load_station_file

import IM_calculation.source_site_dist.src_site_dist as ssd
from qcore.constants import SourceToSiteDist


def write_source_2_site_dists(
    out_file: str,
    stations: np.ndarray,
    locations: np.ndarray,
    r_rup: np.ndarray,
    r_jb: np.ndarray,
    r_x: np.ndarray,
    r_y: np.ndarray,
    backarc: np.ndarray = None,
):
    """Writes the source to site distances to a csv file"""
    data = [locations[:, 0], locations[:, 1], r_rup, r_jb, r_x, r_y]
    cols_names = [
        "lon",
        "lat",
        SourceToSiteDist.R_rup.str_value,
        SourceToSiteDist.R_jb.str_value,
        SourceToSiteDist.R_x.str_value,
        SourceToSiteDist.R_y.str_value,
    ]

    if backarc is not None:
        cols_names.append(SourceToSiteDist.Backarc.str_value)
        data.append(backarc)

    data = np.asarray(data).T

    df = pd.DataFrame(data=data, columns=cols_names, index=stations)
    df.to_csv(out_file, index_label="station")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Script for calculating rrup and rjb for the specified fault and stations"
    )
    parser.add_argument(
        "station_file", type=str, help=".ll file with the stations actual locations"
    )
    parser.add_argument(
        "srf_file", type=str, help="srf file for the source of interest"
    )
    parser.add_argument(
        "-s",
        "--stations",
        nargs="+",
        help="space delimited list of stations names for which "
        "to calculate source to site distances",
        default=None,
    )
    parser.add_argument(
        "-o", "--output", type=str, default="rrups.csv", help="Path to the output file"
    )
    parser.add_argument(
        "-fd",
        "--fd_station_file",
        type=str,
        default=None,
        help="List of stations for a specific domain. Source to site distances "
        "will be calculated for all stations in the fd_station_file "
        "and the station_file",
    )
    parser.add_argument(
        "-b",
        "--backarc",
        action="store_true",
        help="Add a crude definition of back-arc to the output file. 1 if on back-arc, 0 if on forearc",
    )

    args = parser.parse_args()

    # Load the stations
    station_df = load_station_file(args.station_file)

    # Matches the station file with the specified stations or the fd_stat_list. If neither is specified, use the whole list
    station_mask = np.ones(station_df.shape[0], dtype=bool)
    if args.fd_station_file:
        fd_stations_df = load_station_file(args.fd_station_file)
        station_mask = np.isin(station_df.index.values, fd_stations_df.index.values)
    elif args.stations:
        station_mask = np.isin(station_df.index.values, args.stations)

    filtered_station_np = station_df.loc[station_mask].values
    stats_2_calc = station_df.loc[station_mask].index.values

    # Add depth for the stations (hardcoded to 0)
    filtered_station_np = np.concatenate(
        (
            filtered_station_np,
            np.zeros(
                (filtered_station_np.shape[0], 1), dtype=filtered_station_np.dtype
            ),
        ),
        axis=1,
    )

    # Load the srf points
    srf_points = read_srf_points(args.srf_file)

    # Calculate source to site distances
    r_rup, r_jb = ssd.calc_rrup_rjb(srf_points, filtered_station_np)

    plane_info = read_header(args.srf_file, idx=True)

    r_x, r_y = ssd.calc_rx_ry(srf_points, plane_info, filtered_station_np, system="GC2")

    backarc = None
    if args.backarc:
        backarc = ssd.calc_backarc(srf_points, filtered_station_np)

    # Save the result as a csv
    write_source_2_site_dists(
        args.output,
        stats_2_calc,
        filtered_station_np,
        r_rup,
        r_jb,
        r_x=r_x,
        r_y=r_y,
        backarc=backarc,
    )
