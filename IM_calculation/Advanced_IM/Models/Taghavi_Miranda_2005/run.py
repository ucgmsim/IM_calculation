"""
Python script to run a 3D waveform through Taghavi_Miranda_2005 and store the outputs to a txt file
"""

import logging
import os

import numpy as np
import pandas as pd

from IM_calculation.Advanced_IM import runlibs_2d
from IM_calculation.IM.intensity_measures import get_geom
from IM_calculation.IM.Taghavi_Miranda_2005 import Taghavi_Miranda_2005
from qcore.constants import Components
from qcore.timeseries import read_ascii

model_dir = os.path.dirname(__file__)

component_list = [Components.c000, Components.c090]
ALPHA = [0, 1.5, 5, 15, 30]
C = [0.02, 0.05, 0.1]
T = [0.1, 0.4, 0.7, 1.5, 3.0]
STORIES = 10


def main(comp_000, comp_090, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "log")
    logging.basicConfig(
        format="%(asctime)s %(message)s", filename=log_file, level=logging.DEBUG
    )

    waveforms = {}
    waveforms["000"], meta = read_ascii(comp_000, meta=True)
    waveforms["090"] = read_ascii(comp_090)
    dt = meta["dt"]
    results = {}

    for component in component_list:
        results[component.str_value] = {}
        for a in ALPHA:
            for c in C:
                for period in T:
                    im_name = f"TM05_a{a}_c{c}_T{period}_storey"
                    logging.info(f"calculating {im_name}")
                    df = Taghavi_Miranda_2005(
                        waveforms[component.str_value], dt, period, a, c, storey=STORIES
                    )
                    for i in range(STORIES + 1):
                        results[component.str_value][
                            f"{im_name}{i}_disp_peak"
                        ] = df.iloc[i].disp_peak
                        results[component.str_value][
                            f"{im_name}{i}_slope_peak"
                        ] = df.iloc[i].slope_peak
                        results[component.str_value][
                            f"{im_name}{i}_storey_shear_peak"
                        ] = df.iloc[i].storey_shear_peak
                        results[component.str_value][
                            f"{im_name}{i}_total_accel_peak"
                        ] = df.iloc[i].total_accel_peak

    im_csv_fname = os.path.join(output_dir, "Taghavi_Miranda_2005.csv")
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "component"
    geom = pd.Series(
        get_geom(df.loc[Components.c000.str_value], df.loc[Components.c090.str_value]),
        name=Components.cgeom.str_value,
    )
    df = df.append(geom)
    df.to_csv(im_csv_fname)


if __name__ == "__main__":
    args = runlibs_2d.parse_args()
    main(
        getattr(args, Components.c000.str_value),
        getattr(args, Components.c090.str_value),
        args.output_dir,
    )
