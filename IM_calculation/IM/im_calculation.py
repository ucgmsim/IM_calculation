import csv
import glob
import os
import sys
from datetime import datetime
from multiprocessing.pool import Pool
from collections import ChainMap
from typing import List

import numpy as np
import pandas as pd

from qcore import timeseries, constants
from qcore.constants import Components
from qcore.im import order_im_cols_df

from IM_calculation.IM import read_waveform, intensity_measures
from IM_calculation.IM.computeFAS import get_fourier_spectrum


G = 981.0
DEFAULT_IMS = ("PGA", "PGV", "CAV", "AI", "Ds575", "Ds595", "MMI", "pSA")
ALL_IMS = ("PGA", "PGV", "CAV", "AI", "Ds575", "Ds595", "MMI", "pSA", "FAS")

MULTI_VALUE_IMS = ("pSA", "FAS")

FAS_FREQUENCY = np.logspace(-1, 2, num=100, base=10.0)


FILE_TYPE_DICT = {"a": "ascii", "b": "binary"}
META_TYPE_DICT = {"s": "simulated", "o": "observed", "u": "unknown"}
RUNNAME_DEFAULT = "all_station_ims"

OUTPUT_SUBFOLDER = "stations"

MEM_PER_CORE = 7.5e8
MEM_FACTOR = 4


def array_to_dict(value, comps_to_calc, im, comps_to_store):
    """
    convert a numpy arrary that contains calculated im values to a dict {comp: value}
    :param value: calculated intensity measure for a waveform
    :param comps_to_calc: a list of components converted from user input
    :param im:
    :param comps_to_store:user input list of components
    :return: a dict {comp: value}
    """
    value_dict = {}
    # ["090", "ver"], ["090", "000", "geom"], ["090", "000", "ver", "geom"]
    # [0, 2]          [0, 1]                  [0, 1, 2]
    for i in range(value.shape[-1]):
        # pSA returns a 2D array
        if im in MULTI_VALUE_IMS:
            value_dict[comps_to_calc[i].str_value] = value[:, i]
        else:
            value_dict[comps_to_calc[i].str_value] = value[i]
    # In this case, if geom in str_comps,
    # it's guaranteed that 090 and 000 will be present in value_dict
    if Components.cgeom in comps_to_store:
        value_dict[Components.cgeom.str_value] = intensity_measures.get_geom(
            value_dict[Components.c090.str_value], value_dict[Components.c000.str_value]
        )
    # then we pop unwanted keys from value_dict
    for k in comps_to_calc:
        if k not in comps_to_store and k.str_value in value_dict:
            del value_dict[k.str_value]
    return value_dict


def calculate_rotd(spectral_displacements, comps_to_store: List[Components]):
    """
    Calculates rotd for given spectral displacements
    :param spectral_displacements: An array with shape [periods.size, nt, 2] where nt is the number of timesteps in the original waveform
    :param comps_to_store: A list of components to store
    :return: A dictionary with the comps_to_store as keys, and 1d arrays of shape [periods.size] containing the rotd values
    """
    rotd = intensity_measures.calc_rotd(spectral_displacements)
    value_dict = {}
    if Components.crotd50 in comps_to_store:
        value_dict[Components.crotd50] = np.median(rotd, axis=1)
    if Components.crotd100 in comps_to_store:
        value_dict[Components.crotd100] = np.max(rotd, axis=1)
    if Components.crotd100_50 in comps_to_store:
        value_dict[Components.crotd100_50] = np.max(rotd, axis=1) / np.median(
            rotd, axis=1
        )
    return value_dict


def compute_measure_single(
    waveform, ims, comps_to_store, im_options, comps_to_calculate
):
    """
    Compute measures for a single station
    :param: a tuple consisting 5 params: waveform, ims, comp, period, str_comps
    waveform: a single tuple that contains (waveform_acc,waveform_vel)
    :return: {result[station_name]: {[im]: value or (period,value}}
    """
    waveform_acc, waveform_vel = waveform
    DT = waveform_acc.DT
    times = waveform_acc.times

    accelerations = waveform_acc.values

    if waveform_vel is None:
        # integrating g to cm/s
        velocities = timeseries.acc2vel(accelerations, DT) * G
    else:
        velocities = waveform_vel.values

    station_name = waveform_acc.station_name

    result = {(station_name, comp.str_value): {} for comp in comps_to_store}

    def process_single_value_im(im, func, *args, **kwargs):
        if im in ims:
            value = func(*args, **kwargs)
            values_to_store = array_to_dict(
                value, comps_to_calculate, im, comps_to_store
            )
            for comp in comps_to_store:
                if comp.str_value in values_to_store:
                    result[(station_name, comp.str_value)][im] = values_to_store[
                        comp.str_value
                    ]

    process_single_value_im("PGV", intensity_measures.get_max_nd, velocities)
    process_single_value_im("PGA", intensity_measures.get_max_nd, accelerations)
    process_single_value_im(
        "CAV", intensity_measures.get_cumulative_abs_velocity_nd, accelerations, times
    )
    process_single_value_im(
        "AI", intensity_measures.get_arias_intensity_nd, accelerations, G, times
    )
    process_single_value_im("MMI", intensity_measures.calculate_MMI_nd, velocities)
    process_single_value_im(
        "Ds595", intensity_measures.getDs_nd, DT, accelerations, 5, 95
    )
    process_single_value_im(
        "Ds575", intensity_measures.getDs_nd, DT, accelerations, 5, 75
    )

    if "pSA" in ims:
        im = "pSA"
        # store a im type values into a dict {comp: np_array/single float}
        # Geometric is also calculated here
        psa, spectral_displacements = intensity_measures.get_spectral_acceleration_nd(
            accelerations, im_options[im], waveform_acc.NT, DT
        )
        # Store the pSA im values in the format Tuple(List(periods), dict(component: List(im_values)))
        # Where the im_values in the component dictionaries correspond to the periods in the periods list
        pSA_values = array_to_dict(psa, comps_to_calculate, im, comps_to_store)

        if {
            Components.crotd50,
            Components.crotd100,
            Components.crotd100_50,
        }.intersection(comps_to_store):
            # Only run if any of the given components are selected (Non empty intersection)
            rotd = calculate_rotd(spectral_displacements, comps_to_store)
            pSA_values.update(rotd)

        for comp in comps_to_store:
            if comp.str_value in pSA_values:
                for i, val in enumerate(im_options[im]):
                    result[(station_name, comp.str_value)][
                        f"{im}_{str(val)}"
                    ] = pSA_values[comp.str_value][i]

    if "FAS" in ims:
        im = "FAS"
        try:
            value = get_fourier_spectrum(accelerations, DT, im_options[im])
        except FileNotFoundError as e:
            print(
                f"Attempting to compute fourier spectrum raised exception: {e}\nThis was most likely caused by attempting to compute for a waveform with more than 16384 timesteps."
            )
        else:
            values_to_store = array_to_dict(
                value, comps_to_calculate, im, comps_to_store
            )
            for comp in comps_to_store:
                if comp.str_value in values_to_store:
                    for i, val in enumerate(im_options[im]):
                        result[(station_name, comp.str_value)][
                            f"{im}_{str(val)}"
                        ] = values_to_store[comp.str_value][i]

    return result


def get_bbseis(input_path, file_type, selected_stations):
    """
    :param input_path: user input path to bb.bin or a folder containing ascii files
    :param file_type: binary or ascii
    :param selected_stations: list of user input stations
    :return: bbseries, station_names
    """
    bbseries = None
    if file_type == FILE_TYPE_DICT["b"]:
        bbseries = timeseries.BBSeis(input_path)
        if selected_stations is None:
            station_names = bbseries.stations.name
        else:
            station_names = selected_stations
    elif file_type == FILE_TYPE_DICT["a"]:
        search_path = os.path.abspath(os.path.join(input_path, "*"))
        files = glob.glob(search_path)
        station_names = set(map(read_waveform.get_station_name_from_filepath, files))
        if selected_stations is not None:
            station_names = station_names.intersection(selected_stations)
            if len(station_names) == 0:  # empty set
                sys.exit(
                    "could not find specified stations {} in folder {}".format(
                        selected_stations, input_path
                    )
                )
    return bbseries, list(station_names)


def compute_measures_multiprocess(
    input_path,
    file_type,
    wave_type,
    station_names,
    ims=DEFAULT_IMS,
    comp=None,
    im_options=None,
    output=None,
    identifier=None,
    rupture=None,
    run_type=None,
    version=None,
    process=1,
    simple_output=False,
    units="g",
):
    """
    using multiprocesses to computer measures.
    Calls compute_measure_single() to compute measures for a single station
    write results to csvs and an imcalc.info meta data file
    :param input_path:
    :param file_type:
    :param wave_type:
    :param station_names:
    :param ims:
    :param comp:
    :param im_options:
    :param output:
    :param identifier:
    :param rupture:
    :param run_type:
    :param version:
    :param process:
    :param simple_output:
    :return:
    """
    (
        components_to_calculate,
        components_to_store,
    ) = constants.Components.get_comps_to_calc_and_store(comp)

    bbseries, station_names = get_bbseis(input_path, file_type, station_names)

    total_stations = len(station_names)
    steps = get_steps(
        input_path, process, total_stations, "FAS" in ims and bbseries.nt > 32768
    )

    all_results = []
    p = Pool(process)

    i = 0
    while i < total_stations:
        waveforms = read_waveform.read_waveforms(
            input_path,
            bbseries,
            station_names[i : i + steps],
            components_to_calculate,
            wave_type=wave_type,
            file_type=file_type,
            units=units,
        )
        i += steps
        array_params = []
        for waveform in waveforms:
            array_params.append(
                (
                    waveform,
                    sorted(ims),
                    sorted(components_to_store, key=lambda x: x.value),
                    im_options,
                    sorted(components_to_calculate, key=lambda x: x.value),
                )
            )
        all_results.extend(p.starmap(compute_measure_single, array_params))

    all_result_dict = ChainMap(*all_results)
    write_result(all_result_dict, output, identifier, simple_output)

    generate_metadata(output, identifier, rupture, run_type, version)


def get_result_filepath(output_folder, arg_identifier, suffix):
    return os.path.join(output_folder, "{}{}".format(arg_identifier, suffix))


def write_result(result_dict, output_folder, identifier, simple_output):
    """
    write a big csv that contains all calculated im value and single station csvs
    :param result_dict:
    :param output_folder:
    :param identifier: user input run name
    :param simple_output
    :return:output result into csvs
    """
    output_path = get_result_filepath(output_folder, identifier, ".csv")

    results_dataframe = pd.DataFrame.from_dict(result_dict, orient="index")
    results_dataframe.index = pd.MultiIndex.from_tuples(
        results_dataframe.index, names=["station", "component"]
    )
    results_dataframe.sort_values(["station", "component"], inplace=True)
    results_dataframe = order_im_cols_df(results_dataframe)

    # Save the transposed dataframe
    results_dataframe.to_csv(output_path)

    if not simple_output:
        # For each subframe with the same station write it to csv
        for station, sub_frame in results_dataframe.groupby(level=0):
            station_csv = os.path.join(
                output_folder, OUTPUT_SUBFOLDER, "{}_{}.csv".format(identifier, station)
            )
            sub_frame.to_csv(station_csv)


def generate_metadata(output_folder, identifier, rupture, run_type, version):
    """
    write meta data file
    :param output_folder:
    :param identifier: user input
    :param rupture: user input
    :param run_type: user input
    :param version: user input
    :return:
    """
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = get_result_filepath(
        output_folder, identifier, constants.IM_SIM_CALC_INFO_SUFFIX
    )

    with open(output_path, "w") as meta_file:
        meta_writer = csv.writer(meta_file, delimiter=",", quotechar="|")
        meta_writer.writerow(["identifier", "rupture", "type", "date", "version"])
        meta_writer.writerow([identifier, rupture, run_type, date, version])


def validate_input_path(parser, arg_input, arg_file_type):
    """
    validate input path
    :param parser:
    :param arg_input:
    :param arg_file_type:
    :return:
    """
    if not os.path.exists(arg_input):
        parser.error("{} does not exist".format(arg_input))

    if arg_file_type == "b":
        if os.path.isdir(arg_input):
            parser.error(
                "The path should point to a binary file but not a directory. "
                "Correct Sample: /home/tt/BB.bin"
            )
    elif arg_file_type == "a":
        if os.path.isfile(arg_input):
            parser.error(
                "The path should be a directory but not a file. Correct "
                "Sample: /home/tt/sims/"
            )


def validate_period(arg_period, arg_extended_period):
    """
    returns validated user input if pass the validation else raise parser error
    :param parser:
    :param arg_period: input
    :param arg_extended_period: input
    :param im: validated im(s) in a list
    :return: period(s) in a numpy array
    """
    period = np.array(arg_period, dtype="float64")

    if arg_extended_period:
        period = np.unique(np.append(period, constants.EXT_PERIOD))

    return period


def validate_FAS_frequency(arg_fas_freq):
    """
    returns validated user input if pass the validation else raise parser error
    :param arg_fas_freq: input
    :return: frequencies in a numpy array
    """
    frequencies = np.array(arg_fas_freq, dtype="float64")

    return frequencies


def get_steps(input_path, nps, total_stations, high_mem_usage=False):
    """
    :param input_path: user input file/dir path
    :param nps: number of processes
    :param total_stations: total number of stations
    :param high_mem_usage: If a calculation requiring even larger amounts of RAM is required (ie FAS), this increases the estimated RAM even further
    :return: number of stations per iteration/batch
    """
    estimated_mem = os.stat(input_path).st_size * MEM_FACTOR
    if high_mem_usage:
        estimated_mem *= MEM_FACTOR
    available_mem = nps * MEM_PER_CORE
    batches = np.ceil(np.divide(estimated_mem, available_mem))
    steps = int(np.floor(np.divide(total_stations, batches)))
    if steps == 0:
        steps = total_stations
    return steps
