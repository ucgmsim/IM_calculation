import glob
import os
import shutil
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from qcore.constants import Components

from IM_calculation.Advanced_IM import runlibs_2d

BASIC_HORIZONTAL_COMPONENTS = [Components.c000, Components.c090]
IM_NAME = "Spear_3D_Model"
SCRIPT_LOCATION = os.path.dirname(__file__)


def main(c000_filepath, c090_filepath, OpenSees_path, output_dir, im_name, run_script, timeout_threshold):
    os.makedirs(output_dir, exist_ok=True)

    # check for all failed from log
    if runlibs_2d.check_status(output_dir, check_fail=True):
        # Non convergence
        print(f"Model failed to converged in previous run.")
        return

    # check if successfully ran previously
    # skip this component if success
    if runlibs_2d.check_status(output_dir):
        print(f"Model completed in preivous run")
        return

    script = [
        OpenSees_path,
        run_script,
        c000_filepath,
        c090_filepath,
        output_dir,
    ]

    print(" ".join(script))
    # for debug purpose, track the starting and ending time of OpenSees call
    # saves starting time
    runlibs_2d.datetime_to_file(datetime.now(), runlibs_2d.time_type.start_time.name, output_dir)

    for comp in BASIC_HORIZONTAL_COMPONENTS:
        # Give the sub components start times
        os.makedirs(os.path.join(output_dir, comp.str_value), exist_ok=True)
        shutil.copy(os.path.join(output_dir, runlibs_2d.time_type.start_time.name), os.path.join(output_dir, comp.str_value))

    try:
        subprocess.run(script, timeout=timeout_threshold)
    except subprocess.TimeoutExpired:
        # timeouted. save to timed_out instead of end_time
        end_time_type = runlibs_2d.time_type.timed_out.name
    else:
        # save the ending time
        end_time_type = runlibs_2d.time_type.end_time.name

    runlibs_2d.datetime_to_file(datetime.now(), end_time_type, output_dir)

    for comp in BASIC_HORIZONTAL_COMPONENTS:
        # Give the sub components end times
        shutil.copy(os.path.join(output_dir, end_time_type), os.path.join(output_dir, comp.str_value))
        # Copy anaylsis status files into individual components so the status check script can find them
        # Hack for 3d model in a 2d world
        for file in glob.glob(os.path.join(output_dir, "Analysis_*")):
            shutil.copy(file, os.path.join(output_dir, comp.str_value))

    # check for success message after a run
    # marked as failed if any component fail
    station_name = os.path.basename(c000_filepath).split(".")[0]

    if not runlibs_2d.check_status(output_dir):
        print(f"{output_dir} failed to converge for {station_name}.")

        im_csv_failed_name = os.path.join(output_dir, f"{im_name}_failed.csv")

        with open(im_csv_failed_name, "w") as f:
            f.write("status\n")
            f.write("failed")

        for comp in BASIC_HORIZONTAL_COMPONENTS:
            shutil.copy(im_csv_failed_name, os.path.join(output_dir, comp.str_value))

    else:
        for component in BASIC_HORIZONTAL_COMPONENTS:
            create_im_csv(
                output_dir,
                im_name,
                component.str_value,
                os.path.join(output_dir, component.str_value),
                check_converge=False,
            )

        calculate_geom(output_dir, im_name)
        calculate_norm(im_name, output_dir)

        agg_csv(output_dir, im_name)
        print(f"analysis completed for {station_name}")


# calc norm
def read_recorder_all(direction_dir, recorder_name, im_recorder_fname):

    # find corrosponding gravity file
    im_gravity_dir = os.path.join(direction_dir, f"gravity_{recorder_name}")
    im_gravity_recorder = os.path.join(im_gravity_dir, f"gr_{im_recorder_fname}")

    if os.path.exists(im_gravity_recorder):
        gr_value = float(read_out_file(im_gravity_recorder))
    else:
        gr_value = 0

    im_recorder = os.path.join(
        direction_dir, os.path.join(recorder_name, im_recorder_fname)
    )

    with open(im_recorder) as f_im_recorder:
        im_records_list_tmp = [float(line.split()[1]) for line in f_im_recorder]
    im_records_list = []
    for im_record in im_records_list_tmp[:-1]:
        im_record = im_record - gr_value
        im_records_list.append(im_record)

    return im_records_list


def calculate_norm(im_name, output_dir, print_header=True):
    # get 000 and 090 dir
    dir_000 = os.path.join(output_dir, "000")
    dir_090 = os.path.join(output_dir, "090")
    component = "norm"

    im_csv_fname = os.path.join(output_dir, f"{im_name}_{component}.csv")

    value_dict = {}

    # read data from recordings
    for recorder_name in ["disp", "drift", "accl"]:
        # find recorder in 000 first
        # read all 2nd coloumn
        recorder_dir = os.path.join(dir_000, recorder_name)
        im_recorder_glob = os.path.join(recorder_dir, "*.out")
        im_recorders = glob.glob(im_recorder_glob)

        for im_recorder in im_recorders:
            im_recorder_fname = os.path.basename(im_recorder)
            im_recorder_name = os.path.splitext(im_recorder_fname)[0]
            im_record_list = {}

            im_record_list["000"] = read_recorder_all(
                dir_000, recorder_name, im_recorder_fname
            )
            im_record_list["090"] = read_recorder_all(
                dir_090, recorder_name, im_recorder_fname
            )
            im_record_list["norm"] = []
            if len(im_record_list["000"]) != len(im_record_list["090"]):
                raise (
                    ValueError(
                        f"length of 000 and 090 does not match for {im_recorder_fname}"
                    )
                )
            # calulate norm
            for i in range(0, len(im_record_list["000"])):
                norm = np.sqrt(
                    (
                        np.power(im_record_list["000"][i], 2)
                        + np.power(im_record_list["090"][i], 2)
                    )
                )
                im_record_list["norm"].append(norm)

            im_value = max(im_record_list["norm"])
            # value_dict[recorder_name][im_recorder_name] = im_value
            value_dict[im_recorder_name] = im_value

    value_dict = {component: value_dict}
    result_df = pd.DataFrame.from_dict(value_dict, orient="index")
    result_df.index.name = "component"

    cols = list(result_df.columns)
    cols.sort()

    # test if file exist, if exist, no header
    if os.path.isfile(im_csv_fname):
        print_header = False

    result_df.to_csv(im_csv_fname, mode="a", header=print_header, columns=cols)


def calculate_geom(output_dir, im_name):
    """
    generates geom by globing results from 000 and 090
    output_dir: folder that contains adv_im_comp.csv. used to grab data from 000, 090.
    im_name: adv_im model name
    """

    c000_path = os.path.join(output_dir, f"{im_name}_000.csv")
    c090_path = os.path.join(output_dir, f"{im_name}_090.csv")

    df_000 = pd.read_csv(c000_path, dtype={"component": str})
    df_090 = pd.read_csv(c090_path, dtype={"component": str})

    ims = df_000.append(df_090)
    ims.set_index("component", inplace=True)

    if "000" in ims.index and "090" in ims.index:
        line = np.sqrt(ims.loc["090"] * ims.loc["000"])
        line.rename("geom", inplace=True)
        im_geom = pd.DataFrame()
        im_geom = im_geom.append(line)
        im_geom.index.name = "component"
    else:
        raise ValueError(
            f"Error: no 000 or 090 found in csvs: {c000_path}, {c090_path}"
        )
    cols = list(im_geom.columns)
    cols.sort()
    im_csv_fname = os.path.join(output_dir, f"{im_name}_geom.csv")
    im_geom.to_csv(im_csv_fname, columns=cols, index=True, header=True)


def agg_csv(output_dir, im_name, print_header=True):
    """
    aggregate all data into one huge csv that contains multiple rows
    000,090, geom, norm
    """
    # read all df
    df = pd.DataFrame()
    df.index.name = "component"
    for component in ["000", "090", "geom", "norm"]:
        component_csv_fname = os.path.join(output_dir, f"{im_name}_{component}.csv")
        df = df.append(pd.read_csv(component_csv_fname, dtype={"component": str}))
    df.set_index("component", inplace=True)
    cols = list(df.columns)
    cols.sort()
    df.to_csv(
        os.path.join(output_dir, im_name + ".csv"), header=print_header, columns=cols
    )
    # write a csv file


def create_im_csv(
    output_dir,
    im_name,
    component,
    component_dir,
    check_converge=True,
    print_header=True,
    remove_gravity=True,
):
    """
    create a csv file for each single component/analysis
    """
    if check_converge:
        success_glob = os.path.join(component_dir, "Analysis_*")
        success_files = glob.glob(success_glob)
        model_converged = False
        for f in success_files:
            with open(f) as fp:
                contents = fp.read()
            model_converged = model_converged or (contents.strip() == "Successful")
    else:
        # read even if not converged
        model_converged = True

    im_csv_fname = os.path.join(output_dir, f"{im_name}_{component}.csv")
    result_df = pd.DataFrame()
    im_recorder_glob = os.path.join(component_dir, "env*/*.out")
    im_recorders = glob.glob(im_recorder_glob)
    value_dict = {}

    for im_recorder in im_recorders:
        folder_name = os.path.dirname(im_recorder)
        recorder_name = os.path.splitext(os.path.basename(im_recorder))[0]
        im_name = os.path.basename(folder_name)
        # get base name
        # im_type = im_name.split("_")[0]
        im_type = "_".join(im_name.split("_")[1:])
        im_gravity_dir = os.path.join(component_dir, "gravity_" + im_type)
        im_gravity_recorder = os.path.join(
            im_gravity_dir, "gr_" + os.path.basename(im_recorder)
        )

        # find corrosponding gravity file
        #        im_value_tmp = read_out_file(im_recorder, model_converged)
        if os.path.exists(im_gravity_recorder) and remove_gravity:
            gr_value = float(read_out_file(im_gravity_recorder, model_converged))
        #            print(f"{im_recorder} gr_value:{gr_value}")
        else:
            #            print(f'cannot find gr files for {im_gravity_recorder}')
            gr_value = 0
        #        print(f"{im_recorder} gr_value:{gr_value} im_value_tmp:{im_value_tmp} gr_value{gr_value}")
        # im_value = float(im_value_tmp) - float(gr_value)
        with open(im_recorder) as f_im_recorder:
            im_records_list_tmp = [float(line.split()[1]) for line in f_im_recorder]
        # read all lines except the last
        im_records_list = []
        for im_record in im_records_list_tmp[:-1]:
            # loop through all records
            # print(im_record)
            # print(f"im_record: {im_record}")
            im_record = im_record - gr_value
            im_records_list.append(abs(im_record))
        im_value = max(im_records_list)

        #        im_value = read_out_file(im_recorder, model_converged)
        value_dict[recorder_name] = im_value
    value_dict = {component: value_dict}
    result_df = pd.DataFrame.from_dict(value_dict, orient="index")
    result_df.index.name = "component"
    cols = list(result_df.columns)
    cols.sort()
    #    if os.path.isfile(im_csv_fname):
    #        print_header = False
    # result_df = result_df.append(value_dict, ignore_index=True)
    # print(result_df)
    result_df.to_csv(im_csv_fname, header=print_header, columns=cols)


def read_out_file(file, success=True):
    if success:
        with open(file) as f:

            lines = f.readlines()

            value = lines[-1].split()[1]

            return value
    else:
        return float("NaN")


def top_level():

    args = runlibs_2d.parse_args()

    im_name = IM_NAME
    run_script = os.path.join(SCRIPT_LOCATION, "Run_script.tcl")

    c000_component = getattr(args, Components.c000.str_value)
    c090_component = getattr(args, Components.c090.str_value)

    main(c000_component, c090_component, args.OpenSees_path, args.output_dir, im_name, run_script, args.timeout_threshold)


if __name__ == "__main__":
    top_level()
