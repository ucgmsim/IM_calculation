"""
Python script to run a 3D waveform through Taghavi_Miranda_2005 and store the outputs to a txt file
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from IM_calculation.IM.Taghavi_Miranda_2005 import Taghavi_Miranda_2005
from qcore.timeseries import read_ascii

model_dir = os.path.dirname(__file__)

COMPONENT = ["000", "090"]
ALPHA = [0, 1.5, 5, 15, 30]
C = [0.02, 0.05, 0.1]
T = [0.1, 0.4, 0.7, 1.5, 3.0]
STORIES = 10


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "comp_000", help="filepath to a station's 000 waveform ascii file"
    )
    parser.add_argument(
        "comp_090", help="filepath to a station's 090 waveform ascii file"
    )
    parser.add_argument(
        "comp_ver", help="filepath to a station's ver waveform ascii file"
    )
    parser.add_argument(
        "output_dir",
        help="Where the IM_csv file is written to. Also contains the temporary recorders output",
    )

    args, unused_args = parser.parse_known_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_file = os.path.join(args.output_dir, "log")
    logging.basicConfig(
        format="%(asctime)s %(message)s", filename=log_file, level=logging.DEBUG
    )
    logging.debug(f"Ignored arguments: {unused_args}")

    waveforms = {}
    waveforms["000"], meta = read_ascii(args.comp_000, meta=True)
    waveforms["090"] = read_ascii(args.comp_090)
    dt = meta["dt"]
    results = {}

    for component in COMPONENT:
        results[component] = {}
        for a in ALPHA:
            for c in C:
                for period in T:
                    im_name = f"TM05_a{a}_c{c}_T{period}_storey"
                    logging.info(f"calculating {im_name}")
                    df = Taghavi_Miranda_2005(
                        waveforms[component], dt, period, a, c, storey=STORIES
                    )
                    for i in range(STORIES + 1):
                        results[component][im_name + f"{i}_disp_peak"] = df.iloc[
                            i
                        ].disp_peak
                        results[component][im_name + f"{i}_slope_peak"] = df.iloc[
                            i
                        ].slope_peak
                        results[component][
                            im_name + f"{i}_storey_shear_peak"
                        ] = df.iloc[i].storey_shear_peak
                        results[component][im_name + f"{i}_total_accel_peak"] = df.iloc[
                            i
                        ].total_accel_peak

    im_csv_fname = os.path.join(output_dir, "Taghavi_Miranda_2005.csv")
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "component"
    geom = pd.Series(np.sqrt(df.loc["000"] * df.loc["090"]), name="geom")
    df = df.append(geom)
    df.to_csv(im_csv_fname)


if __name__ == "__main__":

    main()
