from pathlib import Path
from collections import namedtuple

import pandas as pd
import yaml
import os
import re
import subprocess
import tempfile

from qcore import timeseries
from IM_calculation.IM.read_waveform import Waveform


advanced_im_dir = os.path.dirname(__file__)
CONFIG_FILE_NAME = os.path.join(advanced_im_dir, "advanced_im_config.yaml")

advanced_im_config = namedtuple(
    "advanced_im_config", ["IM_list", "config_file", "OpenSees_path"]
)
COMP_DICT = {"090": 0, "000": 1, "ver": 2}


def get_config(config_file=CONFIG_FILE_NAME):
    """
    Loads the configuration file and returns the dictionary

    Parameters
    ----------
    config_file : str, optional
        Path to the configuration file
        default: advanced_im_config.yaml
    """
    with open(config_file) as yaml_file:
        loaded_config = yaml.safe_load(yaml_file)
    return loaded_config


def get_im_list(config_file=CONFIG_FILE_NAME):
    """Retrieves the list of IMs that are present in config_file"""
    config = get_config(config_file)
    return list(config.keys())


def compute_ims(accelerations, configuration, adv_im_out_dir):
    """
    Calculates all the Advanced IMs for the given waveforms
    :param accelerations: Acceleration array, 1 column for each component. Ordering is specified in COMP_DICT
    :param configuration: Contains runtime configuration, List of IMs to compute, configuration file containing
                          location of scripts to run, and path to the open sees binary
    :return: None (for now)
    """
    config = get_config(configuration.config_file)
    station_name = accelerations.station_name

    with tempfile.TemporaryDirectory() as f:
        f_dir = Path(f)
        save_waveform_to_tmp_files(f_dir, accelerations, station_name)
        for im in configuration.IM_list:
            out_dir = os.path.join(adv_im_out_dir, im)

            im_config = config[im]
            script = [
                "python",
                os.path.join(advanced_im_dir, im_config["script_location"]),
            ]
            # waveform component sequence
            comp_list = ["000", "090", "ver"]

            script.extend(
                [f_dir / f"{station_name}.{component}" for component in comp_list]
            )
            script.extend([out_dir])

            script.extend(["--OpenSees_path", f"{configuration.OpenSees_path}"])
            # if timeout no None, add timeout
            if type(im_config["timeout_threshold"]) is int:
                script.extend(
                    ["--timeout_threshold", str(im_config["timeout_threshold"])]
                )
            else:
                print("invalid value for timeout_threshold. will use default value.")
            print(" ".join(script))
            subprocess.run(script)


def save_waveform_to_tmp_files(
    tmp_folder: Path, accelerations: Waveform, station_name: str
):
    """
    Writes to the 3 files containing values for all components

    Parameters
    ----------
    tmp_folder : Path
        Folder to save the files
    accelerations : Waveform
        Acceleration waveform with all components
    station_name : str
        Station name
    """
    for component in COMP_DICT.keys():
        filename = tmp_folder / f"{station_name}.{component}"
        timeseries.timeseries_to_text(
            accelerations.values[:, COMP_DICT[component]],
            accelerations.DT,
            filename,
            station_name,
            component,
        )


def read_csv(stations, im_calc_dir, im_type):
    """
    read csv into a pandas dataframe.
    stations: list[(str),] list of station names
    im_calc_dir: IM_calc dir that contains all the stations' result
    im_type: the name of the adv_im model
    """
    # quick check of args format
    assert (
        type(im_type) == str
    ), "im_type should be a string, but get {} instead".format(type(im_type))
    # initial a blank dataframe
    df = pd.DataFrame()

    # loop through all stations
    for station in stations:
        # get csv base on station name
        # use glob(?) and qcore.simulation_structure to get specific station_im.csv
        # TODO: define this structure into qcore.simulation_structure
        im_path = os.path.join(im_calc_dir, station, im_type, f"{im_type}.csv")
        if not os.path.isfile(im_path):
            print(f"{im_path} not found, skipping")
            continue
        # read a df and add station name as colum
        df_tmp = pd.read_csv(im_path)

        # add in the station name before agg
        df_tmp.insert(0, "station", station)

        # append the df
        df = df.append(df_tmp)

    # leave write csv to parent function
    return df


def agg_csv(advanced_im_config, stations, im_calc_dir):
    """
    aggregate and create a csv that contain results from all stations
    """

    adv_im_df_dict = {im: pd.DataFrame() for im in advanced_im_config.IM_list}

    for im_type in advanced_im_config.IM_list:
        adv_im_df_dict[im_type] = adv_im_df_dict[im_type].append(
            read_csv(stations, im_calc_dir, im_type)
        )
        # do a natural sort on the column names
        adv_im_df_dict[im_type] = adv_im_df_dict[im_type][
            list(adv_im_df_dict[im_type].columns[:2])
            + sorted(adv_im_df_dict[im_type].columns[2:], key=natural_key)
        ]
        # check if file exist already, if exist header=False
        adv_im_out = os.path.join(im_calc_dir, f"{im_type}.csv")
        print(f"Dumping adv_im data to : {adv_im_out}")
        if os.path.isfile(adv_im_out):
            print_header = False
        else:
            print_header = True
        adv_im_df_dict[im_type].to_csv(
            adv_im_out, mode="a", header=print_header, index=False
        )


def natural_key(string_):
    """
    using regex to get a sequence of numbers in a string and turn them into int for sorting purpose
    example strings: test_model_v20p1p1  test_model2_v20p1p11
    keys: ["test_model_v", 20, "p", 1, "p", 1],  ["test_model",2,"_v", 20, "p", 1, "p", 11]
    """
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)]
