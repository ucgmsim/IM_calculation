"""
Python script to run a 3D waveform through Steel_MF_5Story and store the outputs to a txt file
"""

import argparse
import glob
import os
import subprocess

import numpy as np
import pandas as pd


DEFAULT_OPEN_SEES_PATH = "OpenSees"

model_dir = os.path.dirname(__file__)


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

    parser.add_argument(
        "--OpenSees_path",
        default=DEFAULT_OPEN_SEES_PATH,
        help="Path to OpenSees binary",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    script = [
        args.OpenSees_path,
        os.path.join(model_dir, "Run_script.tcl"),
        args.comp_000,
        output_dir,
    ]

    print(" ".join(script))

    subprocess.call(script)

    im_name = "Steel_MF_5Story"
    create_im_csv(output_dir, im_name, "000")

    script = [
        args.OpenSees_path,
        os.path.join(model_dir, "Run_script.tcl"),
        args.comp_090,
        output_dir,
    ]

    print(" ".join(script))

    subprocess.call(script)

    create_im_csv(output_dir, im_name, "090", print_header=False)

    im_csv_fname = os.path.join(output_dir, im_name + ".csv")
    calculate_geom(im_csv_fname)


def calculate_geom(im_csv_fname):
    ims = pd.read_csv(im_csv_fname, dtype={"component": str})
    ims.set_index("component", inplace=True)

    if "000" in ims.index and "090" in ims.index:
        line = np.sqrt(ims.loc["090"] * ims.loc["000"])
        line.rename("geom", inplace=True)
        ims = ims.append(line)
    cols = list(ims.columns)
    cols.sort()
    ims.to_csv(im_csv_fname, columns=cols)


def create_im_csv(output_dir, im_name, component, print_header=True):
    """
    After the OpenSees code has run, read the recorder files and output it to a CSV file
    :param output_dir: Path to OpenSees recorders output and CSV path
    :param sub_im_name: IM name that has been calculated. Used for filepath
    :return:
    """
    success_glob = os.path.join(output_dir, "Analysis_*")
    success_files = glob.glob(success_glob)
    model_converged = False
    for f in success_files:
        with open(f) as fp:
            contents = fp.read()
        model_converged = model_converged or (contents.strip() == "Successful")

    im_csv_fname = os.path.join(output_dir, im_name + ".csv")
    im_recorder_glob = os.path.join(output_dir, "env*/*.out")
    im_recorders = glob.glob(im_recorder_glob)
    value_dict = {}

    for im_recorder in im_recorders:
        sub_im_name = os.path.splitext(os.path.basename(im_recorder))[0]
        im_value = read_out_file(im_recorder, model_converged)

        full_im_name = im_name + "_" + sub_im_name
        value_dict[full_im_name] = im_value

    value_dict = {component: value_dict}
    result_df = pd.DataFrame.from_dict(value_dict, orient="index")
    result_df.index.name = "component"
    # print(result_df)

    cols = list(result_df.columns)
    cols.sort()
    result_df.to_csv(im_csv_fname, mode="a", header=print_header, columns=cols)


def read_out_file(file, success=True):
    if success:
        with open(file) as f:

            lines = f.readlines()

            value = lines[-1].split()[1]

            return value
    else:
        return float("NaN")


if __name__ == "__main__":

    main()
