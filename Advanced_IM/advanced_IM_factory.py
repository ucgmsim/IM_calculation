import numpy as np
import os
import subprocess
import tempfile
import yaml
from collections import namedtuple

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
    config = get_config(config_file)
    return list(config.keys())


def compute_ims(accelerations, advanced_im_config):
    config = get_config(advanced_im_config.config_file)
    acc_file = {}
    with tempfile.NamedTemporaryFile() as acc_file[
        "090"
    ], tempfile.NamedTemporaryFile() as acc_file[
        "000"
    ], tempfile.NamedTemporaryFile() as acc_file[
        "ver"
    ]:

        save_waveform_to_tmp_files(acc_file, accelerations)

        for im in advanced_im_config.IM_list:
            im_config = config[im]
            for component in im_config["components"]:
                script = [
                    advanced_im_config.OpenSees_path,
                    os.path.join(advanced_im_dir, im_config["script_location"]),
                    im,
                ]
                script.extend(get_filenames_from_components(acc_file, component))

                print(" ".join(script))
                # subprocess.call(script)


def get_filenames_from_components(acc_file, component):
    return map(lambda x: acc_file[x].name, component)


def save_waveform_to_tmp_files(acc_file, accelerations):
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
