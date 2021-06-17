import argparse
import datetime
from enum import Enum, auto
import glob
import os
import subprocess

import numpy as np
import pandas as pd

from qcore.constants import Components
from IM_calculation.IM.intensity_measures import get_geom

DEFAULT_OPEN_SEES_PATH = "OpenSees"
DF_INDEX_NAME = "component"

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class time_type(Enum):
    start_time = 0
    end_time = 1


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        Components.c000.str_value,
        help="filepath to a station's 000 waveform ascii file",
    )
    parser.add_argument(
        Components.c090.str_value,
        help="filepath to a station's 090 waveform ascii file",
    )
    parser.add_argument(
        Components.cver.str_value,
        help="filepath to a station's ver waveform ascii file",
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


def datetime_to_file(t_value, t_type, out_dir):
    #    print(f'{t_value} {type(t_type)} {t_type}')
    # convert format
    t_value = t_value.strftime(TIME_FORMAT)

    f_name = os.path.join(out_dir, t_type)
    with open(f_name, "w") as f:
        f.write(t_value)


def main(args, im_name, run_script):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    # list of component to run
    # TODO: make this an arg if future model needs vairable list of components
    component_list = [Components.c000, Components.c090]

    model_converged = True
    for component in component_list:
        component_outdir = os.path.join(output_dir, component.str_value)
        # check if folder exist, if not create it (opensees does not create the folder and will crash)
        os.makedirs(component_outdir, exist_ok=True)
        # check if the staion has been finished before.
        # check for all failed from log
        if check_status(component_outdir, check_fail=True):
            print(
                f"{getattr(args,component.str_value)} failed to converged in previous run."
            )
            model_converged = False
            break
        # chech if successfully ran previously
        # skip this component if success
        if check_status(component_outdir):
            print(f"{getattr(args, component.str_value)} completed in preivous run")
            continue

        script = [
            args.OpenSees_path,
            run_script,
            getattr(args, component.str_value),
            component_outdir,
        ]

        print(" ".join(script))
        # for debug purpose, track the starting and ending time of OpenSees call
        # saves starting time
        datetime_to_file(
            datetime.datetime.now(), time_type.start_time.name, component_outdir
        )

        subprocess.run(script)

        # save the ending time
        datetime_to_file(
            datetime.datetime.now(), time_type.end_time.name, component_outdir
        )

        # check for success message after a run
        # marked as failed if any component fail
        if not check_status(component_outdir):
            print(
                f"{component_outdir} failed to converge, skipping rest of the components"
            )
            model_converged = False
            break

    station_name = os.path.basename(getattr(args, component_list[0].str_value)).split(
        "."
    )[0]
    # skip creating csv if any component has zero success count
    if model_converged:
        for component in component_list:
            # special treatment for first component
            # wipe out previous csv to prevent corrupted data
            if component == component_list[0]:
                append_csv = False
            else:
                append_csv = True
            component_outdir = os.path.join(output_dir, component.str_value)
            # aggregate
            create_im_csv(
                output_dir,
                im_name,
                component.str_value,
                component_outdir,
                append_csv=append_csv,
            )

        im_csv_fname = os.path.join(output_dir, f"{im_name}.csv")
        calculate_geom(im_csv_fname)
        print(f"analysis completed for {station_name}")
    else:
        im_csv_failed_name = os.path.join(output_dir, f"{im_name}_failed.csv")
        with open(im_csv_failed_name, "w") as f:
            f.write("status\n")
            f.write("failed")
        print(f"failed to converge for {station_name}")


def check_status(component_outdir, check_fail=False):
    """
    check the status of a run by scanning Analysis_* for keyword: "Successful" / "Failed"
    check_fail: Bools. changes the keyword
    """

    analysis_glob = os.path.join(component_outdir, "Analysis_*")
    analysis_files = glob.glob(analysis_glob)

    if len(analysis_files) == 0:
        return False

    if check_fail:
        keyword = "Failed"
        result = True
        for f in analysis_files:
            with open(f) as fp:
                contents = fp.read()
                result = result and (contents.strip() == keyword)
    else:
        keyword = "Successful"
        result = False
        for f in analysis_files:
            with open(f) as fp:
                contents = fp.read()
                result = result or (contents.strip() == keyword)
    return result


def calculate_geom(im_csv_fname):
    ims = pd.read_csv(im_csv_fname, dtype={DF_INDEX_NAME: str})
    ims.set_index(DF_INDEX_NAME, inplace=True)

    if (
        Components.c000.str_value in ims.index
        and Components.c090.str_value in ims.index
    ):
        line = get_geom(
            ims.loc[Components.c000.str_value], ims.loc[Components.c090.str_value]
        )
        line.rename(Components.cgeom.str_value, inplace=True)
        ims = ims.append(line)
    cols = list(ims.columns)
    cols.sort()
    ims.to_csv(im_csv_fname, columns=cols)


def create_im_csv(output_dir, im_name, component, component_outdir, append_csv=True):
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
        # env_dir = os.path.dirname(im_recorder)
        # env_name = os.path.basename(env_dir).split('_')[-1]
        sub_im_type = sub_im_name.split("_")[0]
        sub_im_gravity_dir = os.path.join(component_outdir, "gravity_" + sub_im_type)
        sub_im_gravity_recorder = os.path.join(
            sub_im_gravity_dir, "gr_" + os.path.basename(im_recorder)
        )
        # find corresponding gravity file
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
    result_df.index.name = DF_INDEX_NAME

    cols = list(result_df.columns)
    cols.sort()
    if append_csv:
        print_header = False
        write_mode = "a"
    else:
        print_header = True
        write_mode = "w"
    result_df.to_csv(im_csv_fname, mode=write_mode, header=print_header, columns=cols)


def read_out_file(file):
    with open(file) as f:

        lines = f.readlines()

        value = lines[-1].split()[1]

        return value
