"""script to quickly check the status of adv_im for a event
checks for existance of aggregated csv
checks for nan/null in the csv
checks station in csv matches stations ran
"""


import argparse
from datetime import datetime
from enum import Enum
import os
import sys

import pandas as pd

from qcore import constants as const
from qcore.formats import load_station_file
from IM_calculation.Advanced_IM.runlibs_2d import check_status, TIME_FORMAT, time_type


class analysis_status(Enum):
    not_started = 0
    finished = 1
    not_converged = 2
    not_finished = 3
    crashed = 4
    unknown = 5


COLUMN_NAMES = [
    "station",
    "model",
    "component",
    "status",
    time_type.start_time.name,
    time_type.end_time.name,
]


def read_timelog(comp_run_dir):
    t_list = [None for y in time_type]
    for t_type in time_type:
        path_logfile = os.path.join(comp_run_dir, t_type.name)
        if not os.path.isfile(path_logfile):
            continue
        with open(path_logfile, "r") as f:
            time = datetime.strptime(f.readline(), TIME_FORMAT)
            t_list[t_type.value] = time
    return t_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "im_calc_dir", type=str, help="The path to the realisation directory"
    )
    parser.add_argument(
        "adv_im_model", nargs="+", type=str, help="list of adv_IM models that ran"
    )

    parser.add_argument(
        "--components",
        default=[const.Components.c000.str_value, const.Components.c090.str_value],
        nargs="+",
        choices=list(const.Components.iterate_str_values()),
        help="list of component that ran",
    )

    parser.add_argument(
        "--station_file",
        default=None,
        type=str,
        help="if set, script will only check for folder-names that match the station file",
    )

    # saves output
    parser.add_argument(
        "--simple_check",
        default=None,
        action="store_true",
        help="a quick comparison of station list, will not generate status.csv",
    )

    args = parser.parse_args()

    return args


def check_log(list_folders, model, components, df_model, break_on_fail=False):
    """
    check of "Failed" msg in logs
    no "Failed" in all Analysis*.txt == crashed
    "Failed" in all Analysis*.txt == failed to converge (normal)
    """
    for station_dir in list_folders:
        # check if folder is a station run folder
        station_model_dir = os.path.join(station_dir, model)
        station_name = os.path.basename(station_dir)
        for comp in components:
            component_outdir = os.path.join(station_model_dir, comp)
            time_list = read_timelog(component_outdir)
            if check_status(component_outdir):
                # success
                station_component_status = analysis_status.finished.value
            elif check_status(component_outdir, check_fail=True):
                # all logs showed "Failed", analysis was unable to converge
                station_component_status = analysis_status.not_converged.value
            elif time_list.count(None) == 0:
                # if start and end are there, but no data = crashed
                station_component_status = analysis_status.crashed.value
            elif (time_list.count(None) == 1) and (
                time_list[time_type.start_time.value] is not None
            ):
                # only start_time exist = wct timed out
                station_component_status = analysis_status.not_finished.value
            elif (time_list.count(None) == 1) and (
                time_list[time_type.start_time.value] is None
            ):
                # something went wrong, only end_time was found
                station_component_status = analysis_status.unknown.value
            else:
                # else not started
                station_component_status = analysis_status.not_started.value
            comp_mask = (df_model["station"] == station_name) & (
                df_model["component"] == comp
            )
            df_model.loc[comp_mask, "status"] = station_component_status
            df_model.loc[comp_mask, time_type.start_time.name] = time_list[
                time_type.start_time.value
            ]
            df_model.loc[comp_mask, time_type.end_time.name] = time_list[
                time_type.end_time.value
            ]
    #    print(f"{df_model}")
    return df_model


def main(im_calc_dir, adv_im_model, components, simple_check=False, station_file=None):

    df_list = []
    for model in adv_im_model:
        csv_path = os.path.join(im_calc_dir, "{}.csv".format(model))

        if station_file is not None:
            station_list = load_station_file(station_file).index.tolist()
        else:
            # glob for station folders
            # station_list = [ y for y in [os.path.join(im_calc_dir, x) for x in os.listdir(im_calc_dir)] if os.path.isdir(y) ]
            station_list = [
                x
                for x in os.listdir(im_calc_dir)
                if os.path.isdir(os.path.join(im_calc_dir, x))
            ]
        list_folders = [os.path.join(im_calc_dir, x) for x in station_list]

        # initialize df with empty value with not started
        df_model = pd.concat(
            [
                pd.DataFrame(
                    {
                        "station": station_list,
                        "model": model,
                        "component": component,
                        "status": analysis_status.not_started.value,
                    },
                    columns=COLUMN_NAMES,
                )
                for component in components
            ],
            ignore_index=True,
        )

        # a quick check to compare station count, will skip all other checks if successful.
        if simple_check:
            # using try/except to prevent crash
            try:
                df_csv = pd.read_csv(csv_path)
            except FileNotFoundError:
                # failed to read a agg csv, leave df_model as is.
                print("csv for {} does not exist".format(model))
                continue
            # check for null/nan
            if df_csv.isnull().values.any():
                # agg csv is there, but value has NaN
                # change all status to unknown
                df_model.loc["status"] = analysis_status.unknown.value
                continue

            # check station count, if match, skip rest of checks
            csv_stations = df_csv.station.unique()
            if len(csv_stations) == len(station_list):
                # station folder count matches csv.station.count
                df_model.loc["status"] = analysis_status.finished.value
                continue
        # check for logs
        check_log(list_folders, model, components, df_model, break_on_fail=True)
        print(model)
        df_list.append((df_model, model))

    #
    result_code = 0b00
    for df, model_name in df_list:
        # check if any status >= 3 or != 0
        if df["status"].ge(analysis_status.not_finished.value).any():
            print(f"{model_name} have errors. Please check the status.csv")
            result_code = result_code | 0b01
        if df["status"].eq(analysis_status.not_started.value).any():
            print(
                f"{model_name} has some stations that havent been analysed. Please check status.csv"
            )
            result_code = result_code | 0b10
        # sort index by status
        df.sort_values("status", inplace=True, ascending=False)
        # map status(int) to string before saving as csv
        df["status"] = df["status"].map(lambda x: analysis_status(x).name)
        status_csv_path = os.path.join(im_calc_dir, "{}_status.csv".format(model_name))
        df.to_csv(status_csv_path, header=True, index=True)
    return result_code


if __name__ == "__main__":
    args = parse_args()
    res = main(
        args.im_calc_dir,
        args.adv_im_model,
        args.components,
        simple_check=args.simple_check,
        station_file=args.station_file,
    )
    sys.exit(res)
