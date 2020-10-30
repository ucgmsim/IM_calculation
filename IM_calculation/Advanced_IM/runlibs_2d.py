import argparse
import glob
import os
import subprocess

import numpy as np
import pandas as pd

DEFAULT_OPEN_SEES_PATH = "OpenSees"


def parse_args():
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

    return args


def main(args, im_name, run_script):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_converged = True
    for component in ["000", "090"]:
        component_outdir = os.path.join(output_dir, component)
        # check if folder exist, if not create it (opensees does not create the folder and will crash)
        os.makedirs(component_outdir, exist_ok=True)
        script = [
            args.OpenSees_path,
            run_script,
            getattr(args, "comp_" + component),
            component_outdir,
        ]

        print(" ".join(script))
        subprocess.call(script)

        # check for success message
        # marked as failed if any component fail
        model_converged = model_converged and check_converge(component_outdir)

    # skip creating csv if any component has zero success count
    if model_converged:
        # aggregate
        create_im_csv(output_dir, im_name, component, component_outdir)

        im_csv_fname = os.path.join(output_dir, im_name + ".csv")
        calculate_geom(im_csv_fname)
    else:
        station_name = os.path.basename(args.comp_000).split(".")[0]
        print(f"failed to converge for {station_name}")


def check_converge(component_outdir):
    success_glob = os.path.join(component_outdir, "Analysis_*")
    success_files = glob.glob(success_glob)
    model_converged = False
    for f in success_files:
        with open(f) as fp:
            contents = fp.read()
        model_converged = model_converged or (contents.strip() == "Successful")
    return model_converged


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


def create_im_csv(output_dir, im_name, component, component_outdir, print_header=True):
    """
    After the OpenSees code has run, read the recorder files and output it to a CSV file
    :param output_dir: Path to OpenSees recorders output and CSV path
    :param sub_im_name: IM name that has been calculated. Used for filepath
    :return:
    """

    im_csv_fname = os.path.join(output_dir, im_name + ".csv")
    im_recorder_glob = os.path.join(component_outdir, "env*/*.out")
    im_recorders = glob.glob(im_recorder_glob)
    value_dict = {}

    for im_recorder in im_recorders:
        sub_im_name = os.path.splitext(os.path.basename(im_recorder))[0]

        # get base name
        sub_im_type = sub_im_name.split("_")[0]
        sub_im_gravity_dir = os.path.join(component_outdir, "gravity_" + sub_im_type)
        sub_im_gravity_recorder = os.path.join(
            sub_im_gravity_dir, "gr_" + os.path.basename(im_recorder)
        )
        # find corrosponding gravity file
        if os.path.exists(sub_im_gravity_recorder):
            gr_value = float(read_out_file(sub_im_gravity_recorder))
        else:
            gr_value = 0

        # read the whole csv instead of just last line
        with open(im_recorder) as f_im_recorder:
            im_records_list_tmp = [float(line.split()[1]) for line in f_im_recorder]
        im_records_list = []
        # read all lines except the last
        for im_record in im_records_list_tmp[:-1]:
            # loop through all records
            im_record = im_record - gr_value
            im_records_list.append(abs(im_record))
        im_value = max(im_records_list)

        full_im_name = im_name + "_" + sub_im_name
        value_dict[full_im_name] = im_value

    value_dict = {component: value_dict}
    result_df = pd.DataFrame.from_dict(value_dict, orient="index")
    result_df.index.name = "component"

    cols = list(result_df.columns)
    cols.sort()
    # test if file exist, if exist, no header
    if os.path.isfile(im_csv_fname):
        print_header = False
    result_df.to_csv(im_csv_fname, mode="a", header=print_header, columns=cols)


def read_out_file(file):
    with open(file) as f:

        lines = f.readlines()

        value = lines[-1].split()[1]

        return value
