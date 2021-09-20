from pathlib import Path
import argparse
import glob
import sys

import numpy as np
import pandas as pd

from qcore.formats import load_im_file_pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute logarithmic mean/median of IM values across realisations of a given fault"
    )
    parser.add_argument(
        "csv_path",
        help="path to directory containing CSV files to recursively search",
        type=Path,
    )  # if set too deep, search will be slow.
    parser.add_argument(
        "--faults",
        nargs="+",
        type=str,
        help="List of faults to process, separated by spaces.",
        required=True,
    )
    parser.add_argument(
        "--median",
        dest="stat_fn",
        action="store_const",
        const="median",
        default="mean",
        help="find median (default: mean)",
    )
    parser.add_argument(
        "--im",
        dest="im_types",
        nargs="+",
        type=str,
        help="list of IM types. all chosen if unspecified",
    )  # eg. --im PGV PGA
    parser.add_argument(
        "--output",
        dest="output_dir",
        help="path for the output CSV file to be saved",
        default=Path.cwd(),
        type=Path,
    )

    args = parser.parse_args()

    csv_path = args.csv_path.absolute()

    faults = args.faults
    stat_fn = args.stat_fn
    im_types = args.im_types
    output_dir = args.output_dir.absolute()

    if csv_path.exists() and csv_path.is_dir():
        print("Checked: CSV search directory {}".format(csv_path))
    else:
        print("Error: invalid path : {}".format(csv_path))
        sys.exit(0)

    for fault_name in faults:
        im_csv_paths = glob.glob(
            "{}/**/{}_REL*.csv".format(csv_path, fault_name), recursive=True
        )
        im_csv_paths.sort()

        if len(im_csv_paths) > 0:
            print("Checked: IM csv files located")
        else:
            print("Error: no IM csv files found")
            sys.exit(0)

        print(im_csv_paths)

        if output_dir.exists() and output_dir.is_dir():
            print("Checked: Output directory {}".format(output_dir))
        else:
            Path.mkdir(output_dir)
            print("Created: Output directory {}".format(output_dir))

        rel_im_dfs = []
        for c in im_csv_paths:
            df = load_im_file_pd(c)
            if im_types:
                assert set(df.columns).issuperset(im_types), (
                    f"The following IMs aren't present in the IM csv: {', '.join(set(im_types).difference(df.columns))}"
                    f"Available IMs: {', '.join(df.columns)}"
                )
                rel_im_dfs.append(df[im_types])
            else:
                rel_im_dfs.append(df)

        print(
            "Summarising IM values at {} stations from {} realisations for IM types: {}".format(
                len(rel_im_dfs[0].index.unique(0)),
                len(rel_im_dfs),
                im_types if im_types else "All",
            )
        )

        merged_im_df = pd.concat(rel_im_dfs, axis=0, keys=range(len(rel_im_dfs)))
        if stat_fn == "mean":
            log_mean_im = np.exp(np.log(merged_im_df).mean(level=[1, 2]))
        else:
            log_mean_im = np.exp(np.log(merged_im_df).median(level=[1, 2]))
        log_stdev_im = np.log(merged_im_df).std(level=[1, 2], ddof=0)

        log_stdev_im.columns = [f"{im}_sigma" for im in log_stdev_im.columns]

        summary_df = pd.merge(
            log_mean_im, log_stdev_im, left_index=True, right_index=True
        )

        output_file = output_dir / f"{fault_name}_log_{stat_fn}.csv"

        summary_df.to_csv(output_file)
        print("Completed...Written {}".format(output_file))
