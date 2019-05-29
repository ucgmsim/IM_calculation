#!/usr/bin/env python3
"""Script for calculating rrup and rjb for the specified fault and stations"""
import argparse
import numpy as np
import pandas as pd

from qcore.srf import read_srf_points
from qcore.formats import load_station_file
from qcore.constants import SourceToSiteDist
from source_site_dist.src_site_dist_calc import calc_rrup_rjb


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Script for calculating rrup and rjb for the " "specified fault and stations"
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

    args = parser.parse_args()

    # Load the stations
    station_df = load_station_file(args.station_file)

    # Get the location for all station in the fd station file
    if args.fd_station_file:
        fd_stations_df = load_station_file(args.fd_station_file)
        matched_df = fd_stations_df.join(station_df, how="inner", lsuffix="_fd")
        locs_2_calc = matched_df.loc[:, ("lon", "lat")].values
        stats_2_calc = matched_df.index.values
    elif args.stations:
        locs_2_calc = station_df.loc[[args.stations], "lon", "lat"].values
        stats_2_calc = np.asarray(args.stations)
    else:
        raise argparse.ArgumentError(
            "Either --fd_station_file or --stations has to be set"
        )

    # Add depth for the stations (hardcoded to 0)
    locs_2_calc = np.concatenate(
        (locs_2_calc, np.zeros((locs_2_calc.shape[0], 1), dtype=locs_2_calc.dtype)),
        axis=1,
    )

    # Load the srf points
    srf_points = read_srf_points(args.srf_file)

    # Calculate source to site distances
    r_rup, r_jb = calc_rrup_rjb(srf_points, locs_2_calc)

    # Save the result as a csv
    write_source_2_site_dists(args.output, stats_2_calc, locs_2_calc, r_rup, r_jb)
