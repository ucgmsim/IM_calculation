import argparse
import glob
import os
import pandas as pd

from IM_calculation.Advanced_IM import runlibs_2d


IM_NAME = "SDOF_IMK_rotd"
DEFAULT_OPEN_SEES_PATH = "OpenSees"

model_dir = os.path.dirname(__file__)

SCRIPT_LOCATION = os.path.dirname(__file__)

def main():

    args = runlibs_2d.parse_args()
    im_name = IM_NAME
    run_script = os.path.join(SCRIPT_LOCATION, "run_script.tcl")
    runlibs_2d.main(args, im_name, run_script)

if __name__ == "__main__":

    main()


