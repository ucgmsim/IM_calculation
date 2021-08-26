"""
Python script to run a 3D waveform through Haselton_RC_12Story and store the outputs to a txt file
"""

import os

from IM_calculation.Advanced_IM import runlibs_2d

SCRIPT_LOCATION = os.path.dirname(__file__)
IM_NAME = "Haselton_RC_12Story"


def main():

    args = runlibs_2d.parse_args()
    im_name = IM_NAME
    run_script = os.path.join(SCRIPT_LOCATION, "Run_script.tcl")

    runlibs_2d.main(args, im_name, run_script)


if __name__ == "__main__":

    main()
