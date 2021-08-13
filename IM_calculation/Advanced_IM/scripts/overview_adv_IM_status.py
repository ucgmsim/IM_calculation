#!/usr/bin/env python
# coding: utf-8

import argparse
from pathlib import Path
import sys

import pandas as pd

from qcore import simulation_structure

ALL_STATUS = [
    "finished",
    "not_started",
    "not_converged",
    "not_finished",
    "timed_out",
    "crashed",
    "unknown",
]
TYPES = ["sim", "obs", "custom"]

PATTERN_TEMPLATE = {
    "sim": simulation_structure.get_im_calc_dir("", "*"),
    "obs": "ObservedGroundMotions/IM_calc/*",
}


def aggregate_status_files(type, model, sim_root, outdir, pattern=None, verbose=False):
    search_pattern = PATTERN_TEMPLATE.get(type, pattern) + f"/{model}_status.csv"
    print(f"Searching for {sim_root}/{search_pattern}")

    event_status_files = list(sim_root.glob(search_pattern))

    if len(event_status_files) == 0:
        print(f"Error:No status files found at {sim_root}/{search_pattern}")
        print(f"Check the sim_root, model and search pattern")
        sys.exit(0)

    event_status_dict = {}
    for event_status_file in event_status_files:
        if type == "sim":
            event_name = Path(event_status_file).parents[2].name
        else:
            event_name = Path(event_status_file).parent.name
        event_status_df = pd.read_csv(event_status_file)
        event_status_dict[event_name] = {"total": 0}
        for status in ALL_STATUS:
            count = event_status_df["status"].value_counts().get(status, 0)
            event_status_dict[event_name][status] = count
            event_status_dict[event_name]["total"] += count

    overall_status_df = pd.DataFrame.from_dict(
        event_status_dict, orient="index", columns=["total"] + ALL_STATUS
    )

    overall_status_df.to_csv(outdir / f"overall_status_{model}_{type}.csv")

    not_finished_df = overall_status_df.loc[
        overall_status_df["total"] != overall_status_df["finished"]
    ]
    print("------------")
    print(
        overall_status_df.sum(axis=0).to_string(header=None)
    )  # .to_string(header=None) removes dtype: int64
    print("------------")
    if len(not_finished_df) > 0:
        print(
            f"{len(not_finished_df)} incomplete/unsuccessful event(s) found (Total: {len(overall_status_df)})"
        )
        if verbose:
            print(not_finished_df)
    else:
        print(f"All completed successfully! (Total: {len(overall_status_df)})")
    return overall_status_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sim_root",
        type=Path,
        help="Root path of GM sim.ie. Parent directory of Runs or ObservedGroundMotions",
    )
    parser.add_argument("model", help="Name of the advanced IM model")
    parser.add_argument(
        "type",
        choices=TYPES,
        default="custom",
        help="Type of data. If neither sim or obs, choose custom and provide --pattern",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path.cwd(), help="Output directory"
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help='Status file search pattern for custom type. eg.) "Runs/*/*/IM_calc"',
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print unfinished events summary",
    )

    args = parser.parse_args()

    assert args.sim_root.is_dir(), "Provide a valid directory path"

    if args.type == "custom":
        assert args.pattern is not None, "Type custom must have a search pattern"
        args.pattern = args.pattern.strip(
            "/"
        )  # remove leading or trailing "/" (if any)
    else:
        if args.pattern is not None:
            print(
                f"Warning: Type {args.type} uses pre-defined search pattern - supplied one ignored"
            )

    aggregate_status_files(
        args.type, args.model, args.sim_root, args.outdir, args.pattern, args.verbose
    )


if __name__ == "__main__":
    main()
