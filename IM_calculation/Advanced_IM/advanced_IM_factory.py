from collections import namedtuple
import numpy as np
import os
import subprocess
import tempfile
import yaml

from qcore.timeseries import seis2txt

advanced_im_dir = os.path.dirname(__file__)
CONFIG_FILE_NAME = os.path.join(advanced_im_dir, "advanced_im_config.yaml")
VALUES_PER_LINE = 6

advanced_im_config = namedtuple(
    "advanced_im_config", ["IM_list", "config_file", "OpenSees_path"]
)
COMP_DICT = {"090": 0, "000": 1, "ver": 2}


def get_config(config_file=CONFIG_FILE_NAME):
    with open(config_file) as cf:
        config = yaml.safe_load(cf)
    return config


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
        f_dir = os.path.join(f, "")  # ensure has a trailing slash on filename
        save_waveform_to_tmp_files(f_dir, accelerations, station_name)

        for im in configuration.IM_list:
            out_dir = os.path.join(adv_im_out_dir, im)

            im_config = config[im]
            components = ["000", "090", "ver"]
            script = [
                "python",
                os.path.join(advanced_im_dir, im_config["script_location"]),
                "--OpenSees_path",
                configuration.OpenSees_path,
            ]
            script.extend(
                [get_acc_filename(f_dir, station_name, x) for x in components]
            )
            script.extend([out_dir])

            print(" ".join(script))
            subprocess.call(script)


def get_acc_filename(folder, stat, component):
    # returns the same filename structure as timeseries.seis2txt
    return "%s%s.%s" % (folder, stat, component)


def save_waveform_to_tmp_files(tmp_folder, accelerations, station_name):
    """
    Writes to the 3 files containing values for all components
    :param acc_file: Dict containing file handles for each component specified
    :param accelerations: Acceleration array, 1 column for each component. Ordering is specified in COMP_DICT
    :return: None
    """
    # TODO: Fix bug when differing components are specified in calculate_ims
    for component in COMP_DICT.keys():
        seis2txt(
            accelerations.values[:,COMP_DICT[component]], accelerations.DT, tmp_folder, station_name, component
        )
