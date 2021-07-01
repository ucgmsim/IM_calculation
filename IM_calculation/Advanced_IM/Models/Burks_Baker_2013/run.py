"""
Python script to run a 3D waveform through Taghavi_Miranda_2005 and store the outputs to a txt file
"""

import logging
import os

import numpy as np
import pandas as pd

from IM_calculation.Advanced_IM import runlibs_2d
from IM_calculation.IM.intensity_measures import get_geom
from IM_calculation.IM.Burks_Baker_2013_elastic_inelastic import Bilinear_Newmark_withTH
from IM_calculation.IM import read_waveform, intensity_measures

from qcore.timeseries import read_ascii
from qcore import constants
from qcore.im import order_im_cols_df

model_dir = os.path.dirname(__file__)

COMPONENT = ["090", "000"]

STORIES = 10

z_list = [0.05]  # damping ratio
alpha_list = [0.05]  # strain hardening ratio
dy_list = [0.1765]  # strain hardening ratio
dt_list = [0.005]
period = np.array(constants.DEFAULT_PSA_PERIODS)


def main(comp_000, comp_090, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "log")
    logging.basicConfig(
        format="%(asctime)s %(message)s", filename=log_file, level=logging.DEBUG
    )

    waveforms = {}
    waveforms["000"], meta = read_ascii(comp_000, meta=True)
    waveforms["090"] = read_ascii(comp_090)
    DT = meta["dt"]
    NT = meta["nt"]

    results = {}

    accelerations = np.array((waveforms["090"], waveforms["000"])).T
    all_im_names = []
    for z in z_list:
        for alpha in alpha_list:
            for dy in dy_list:
                for dt in dt_list:
                    displacements = intensity_measures.get_SDI_nd(
                        accelerations, period, NT, DT, z, alpha, dy, dt
                    )
                    sdi_values = np.max(np.abs(displacements), axis=1)
                    im_names = ["SDI_{}".format(t) for t in period]
                    all_im_names.extend(im_names)
                    for i, component in enumerate(COMPONENT):
                        results[component] = {}
                        for j, im_name in enumerate(im_names):
                            results[component][im_name] = sdi_values[j, i] * 100

    im_csv_fname = os.path.join(output_dir, "Burks_Baker_2013.csv")
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "component"
    geom = pd.Series(get_geom(df.loc["000"], df.loc["090"]), name="geom")
    df = df.append(geom)

    df[all_im_names].to_csv(im_csv_fname)


if __name__ == "__main__":
    args = runlibs_2d.parse_args()
    main(args.comp_000, args.comp_090, args.output_dir)
