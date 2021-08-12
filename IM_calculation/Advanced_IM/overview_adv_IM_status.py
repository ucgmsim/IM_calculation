#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import pandas as pd
from pathlib import Path
import sys

ALL_STATUS = [
    "finished",
    "not_started",
    "not_converged",
    "not_finished",
    "timed_out",
    "crashed",
    "unknown",
]
TYPES = ["obs", "sim"]

SIM_DIR_SUFFIX = "Runs/*/*/IM_calc/{}_status.csv"
OBS_DIR_SUFFIX = "ObservedGroundMotions/IM_calc/*/{}_status.csv"


def process(type, model, path, outdir):
    event_status_files = glob.glob(str(path))

    if len(event_status_files) == 0:
        print(f"Error:No status files found at {path}")
        print(f"Check the root_path and model")
        sys.exit(0)

    event_status_dict = {}
    for event_status_file in event_status_files:
        if type == "sim":
            event_name = Path(event_status_file).parents[2].name
        else:
            event_name = Path(event_status_file).parent.name
        event_status_df = pd.read_csv(event_status_file)
        event_status_dict[event_name] = {"total": 0}
        for st in ALL_STATUS:
            try:
                count = event_status_df["status"].value_counts()[st]
            except KeyError:
                count = 0
            event_status_dict[event_name][st] = count
            event_status_dict[event_name]["total"] += count

    overall_status_df = pd.DataFrame.from_dict(
        event_status_dict, orient="index", columns=["total"] + ALL_STATUS
    )
    overall_status_df.to_csv(outdir / f"overall_status_{model}_{type}.csv")
    print(overall_status_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_path",
        type=Path,
        help="Root path of GM sim.ie. Parent directory of Runs or ObservedGroundMotiohs",
    )
    parser.add_argument("model", help="Name of the advanced IM model")
    parser.add_argument("type", choices=TYPES, help="Observation or Simulation")
    parser.add_argument(
        "--outdir", type=Path, default=Path.cwd(), help="Output directory"
    )

    args = parser.parse_args()

    if not args.root_path.is_dir():
        raise AssertionError("Provide a valid directory path")

    if args.type == "sim":
        data_path = args.root_path / SIM_DIR_SUFFIX.format(args.model)
    else:
        data_path = args.root_path / OBS_DIR_SUFFIX.format(args.model)

    process(args.type, args.model, data_path, args.outdir)
