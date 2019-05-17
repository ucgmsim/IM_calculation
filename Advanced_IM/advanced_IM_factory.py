from collections import namedtuple
import numpy as np
import os
import subprocess
import tempfile
import yaml

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


def compute_ims(accelerations, configuration):
    """
    Calculates all the Advanced IMs for the given waveforms
    :param accelerations: Acceleration array, 1 column for each component. Ordering is specified in COMP_DICT
    :param configuration: Contains runtime configuration, List of IMs to compute, configuration file containing
                          location of scripts to run, and path to the open sees binary
    :return: None (for now)
    """
    config = get_config(configuration.config_file)
    acc_file = {}
    with tempfile.NamedTemporaryFile() as acc_file[
        "090"
    ], tempfile.NamedTemporaryFile() as acc_file[
        "000"
    ], tempfile.NamedTemporaryFile() as acc_file[
        "ver"
    ]:

        save_waveform_to_tmp_files(acc_file, accelerations)

        for im in configuration.IM_list:
            im_config = config[im]
            for components in im_config["components"]:
                script = [
                    configuration.OpenSees_path,
                    os.path.join(advanced_im_dir, im_config["script_location"]),
                    im,
                ]
                script.extend([acc_file[x].name for x in components])

                print(" ".join(script))
                subprocess.call(script)


def save_waveform_to_tmp_files(acc_file, accelerations):
    """
    Writes to the 3 files specified in acc_file all components
    :param acc_file: Dict containing file handles for each component specified
    :param accelerations: Acceleration array, 1 column for each component. Ordering is specified in COMP_DICT
    :return: None
    """
    # TODO: Fix bug when differing components are specified in calculate_ims
    for component in COMP_DICT.keys():
        nt = accelerations.shape[0]
        divisible = nt - nt % VALUES_PER_LINE
        np.savetxt(
            acc_file[component],
            accelerations[:divisible, COMP_DICT[component]].reshape(
                -1, VALUES_PER_LINE
            ),
            fmt="%13.5e",
        )
        np.savetxt(
            acc_file[component],
            accelerations[divisible:, COMP_DICT[component]],
            fmt="%13.5e",
        )
