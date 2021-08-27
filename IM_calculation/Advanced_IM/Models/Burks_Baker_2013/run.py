"""
Python script to run a 3D waveform through Burks_Baker_2013_elastic_inelastic and store the outputs to a txt file
"""

import argparse

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from IM_calculation.Advanced_IM import runlibs_2d
from IM_calculation.IM.intensity_measures import get_geom
from IM_calculation.IM import intensity_measures
from IM_calculation.IM import im_calculation

from qcore.timeseries import read_ascii
from qcore import constants
from qcore.constants import Components

component_list = [Components.c000, Components.c090]
rotd_comps = [Components.crotd50, Components.crotd100, Components.crotd100_50]
STORIES = 10

z_list = [0.05]  # damping ratio
alpha_list = [0.05]  # strain hardening ratio
dy_list = [0.1765, 0.4, 0.6]  # strain hardening ratio
dt_list = [0.005]
period = np.array(constants.DEFAULT_PSA_PERIODS)

IM_NAME = Path(__file__).parent.name


def main(comp_000: Path, comp_090: Path, rotd: bool, output_dir: Path):
    """

    Parameters
    ----------
    comp_000 : Path to 000 component waveform file
    comp_090 : Path to 090 component waveform file
    rotd : True if rotd components need to be computed
    output_dir : Path to the output directory
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "log"
    logging.basicConfig(
        format="%(asctime)s %(message)s", filename=log_file, level=logging.DEBUG
    )

    waveforms = {}
    waveforms["000"], meta = read_ascii(comp_000, meta=True)
    waveforms["090"] = read_ascii(comp_090)
    DT = np.float32(meta["dt"])  # to match the behaviour of basic SDI
    NT = meta["nt"]

    results = {}

    accelerations = np.array((waveforms["000"], waveforms["090"])).T
    ordered_columns_dict = {}

    for z in z_list:
        for dt in dt_list:
            for alpha in alpha_list:
                for dy in dy_list:
                    displacements = (
                        intensity_measures.get_SDI_nd(
                            accelerations, period, NT, DT, z, alpha, dy, dt
                        )
                        * 100  # Burks & Baker returns m, but output is stored in cm
                    )
                    sdi_values = np.max(np.abs(displacements), axis=1)
                    im_names = []
                    for t in period:
                        im_name = f"SDI_{t}_dy{dy}_a{alpha}_dt{dt}_z{z}"
                        im_names.append(im_name)
                        # for each period, we will have a list of im_names varying dy,alpha,dt and z, to help ordering columns later
                        if t not in ordered_columns_dict:
                            ordered_columns_dict[t] = []
                        ordered_columns_dict[t].append(im_name)

                    for i, component in enumerate(component_list):
                        if component.str_value not in results:
                            results[component.str_value] = {}
                        for j, im_name in enumerate(im_names):
                            results[component.str_value][im_name] = sdi_values[j, i]
                    if not rotd:
                        continue
                    rotd_values = im_calculation.calculate_rotd(
                        displacements, rotd_comps
                    )
                    for component in rotd_comps:
                        if component.str_value not in results:
                            results[component.str_value] = {}
                        for j, im_name in enumerate(im_names):
                            results[component.str_value][im_name] = rotd_values[
                                component.str_value
                            ][j]

    ordered_columns = []
    for t in period:
        ordered_columns.extend(
            ordered_columns_dict[t]
        )  # order by period first, then by dy, alpha, dt and z

    im_csv_fname = output_dir / f"{IM_NAME}.csv"
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "component"
    geom = pd.Series(
        get_geom(df.loc[Components.c000.str_value], df.loc[Components.c090.str_value]),
        name=Components.cgeom.str_value,
    )

    df = df.append(geom)
    new_index = component_list + [Components.cgeom]
    if rotd:
        new_index += rotd_comps
    df = df.reindex([c.str_value for c in new_index])
    df[ordered_columns].to_csv(im_csv_fname)


def parse_args():
    # extended switch returns parser to allow extra arguments to be added
    parser = runlibs_2d.parse_args(extended=True)
    parser.add_argument(
        "--norotd",
        action="store_false",
        default=True,
        dest="rotd",
        help="skip computing rotd component",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.ver is not None:
        print(f"Info: Component ver is ignored: {args.ver}")
    main(
        Path(getattr(args, Components.c000.str_value)),
        Path(getattr(args, Components.c090.str_value)),
        args.rotd,
        Path(args.output_dir),
    )
