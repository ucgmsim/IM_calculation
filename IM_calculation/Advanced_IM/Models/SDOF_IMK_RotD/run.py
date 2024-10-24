
import os
import pandas as pd
from pathlib import Path

from IM_calculation.Advanced_IM import runlibs_2d
from IM_calculation.IM import im_calculation

from qcore.constants import Components


IM_NAME = "SDOF_IMK_RotD"
DEFAULT_OPEN_SEES_PATH = "OpenSees"

model_dir = os.path.dirname(__file__)

SCRIPT_LOCATION = os.path.dirname(__file__)

component_list = [Components.c000, Components.c090]
rotd_comps = [Components.crotd50, Components.crotd100, Components.crotd100_50]


def main():

    args = runlibs_2d.parse_args()

    run_script = os.path.join(SCRIPT_LOCATION, "Run_script.tcl")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)


    results = {}

    model_converged = runlibs_2d.run_and_check(run_script,
                                               [getattr(args, component_list[0].str_value),getattr(args, component_list[1].str_value)],
                                               str(output_dir),args.OpenSees_path, args.timeout_threshold)

    assert model_converged, f"{output_dir} failed to converge."

    with open(os.path.join(output_dir, "max_rotD_resp.txt"), "r") as f:
        data = f.readlines()
    rotd_values = im_calculation.get_rotd_components_dict([float(x) for x in data[0].split()], rotd_comps)

    for component in rotd_comps:
        if component.str_value not in results:
            results[component.str_value] = {}

            results[component.str_value][IM_NAME] = rotd_values[
                component.str_value
            ]

    im_csv_fname = output_dir / f"{IM_NAME}.csv"
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "component"

    df = df.reindex([c.str_value for c in rotd_comps])
    df[[IM_NAME]].to_csv(im_csv_fname) # df[[IM_NAME]] (rather than df) enforces the output in the same order as rotd_comps list

if __name__ == "__main__":
    main()


