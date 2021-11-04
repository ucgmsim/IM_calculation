import csv
import glob
import os
import sys
from datetime import datetime
from functools import partial
from multiprocessing.pool import Pool
from collections import ChainMap
from typing import List, Iterable

import numpy as np
import pandas as pd

from qcore import timeseries, constants, shared
from qcore.constants import Components
from qcore.im import order_im_cols_df
from qcore.qclogging import get_basic_logger
from qcore.progress_tracker import ProgressTracker
from IM_calculation.Advanced_IM import advanced_IM_factory
from IM_calculation.IM import read_waveform, intensity_measures
from IM_calculation.IM.intensity_measures import G
from IM_calculation.IM.computeFAS import get_fourier_spectrum


DEFAULT_IMS = ("PGA", "PGV", "CAV", "AI", "Ds575", "Ds595", "MMI", "pSA")
ALL_IMS = (
    "PGA",
    "PGV",
    "CAV",
    "AI",
    "Ds575",
    "Ds595",
    "MMI",
    "pSA",
    "SED",
    "FAS",
    "SDI",
)


MULTI_VALUE_IMS = ("pSA", "FAS", "SDI")

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


def check_rotd(comps_to_store: Iterable[Components]) -> bool:
    """
    Checks for any rotd components in the components to store list
    :param comps_to_store: An iterable of the Components enum
    :return: True if any rotd components are to be stored, false otherwise
    """
    return bool(
        {Components.crotd50, Components.crotd100, Components.crotd100_50}.intersection(
            comps_to_store
        )
    )


def calculate_rotd(
    accelerations,
    comps_to_store: List[Components],
    func=lambda x: np.max(np.abs(x), axis=-2),
):
    """
    Calculates rotd for given accelerations

    Multiplying the waveforms by the rotation matrices gives an array with shape either
    (periods, nt, n_rotations) (for pSA) or (nt, n_rotations) (for single dimension IMs).
    The max is taken over the second axis from the rear (nt),
    so for pSA we get an array with shape (periods, n_rotations), while single dimension IMs gets (n_rotations)
    Which is then taken the median for RotD50 and the maximum for RotD100

    :param accelerations: An array with shape [[periods.size,] nt, 2]
        where the first axis is optional and if present is equal to the number of periods in the intensity measure.
        nt is the number of timesteps in the original waveform
    :param comps_to_store: A list of components to store
    :param func: The function to apply to the rotated waveforms. Defaults to taking the maximum absolute value across all rotations (used by PGA, PGV, pSA)
    :return: A dictionary with the comps_to_store as keys, and 1d arrays of shape [periods.size] containing the rotd values
    """
    # Selects the first two basic components. get_comps_to_calc_and_store makes sure that the first two are 000 and 090
    rotd = intensity_measures.get_rotations(accelerations[..., [0, 1]], func=func)
    value_dict = {}

    rotd50 = np.median(rotd, axis=-1)
    rotd100 = np.max(rotd, axis=-1)

    if Components.crotd50 in comps_to_store:
        value_dict[Components.crotd50.str_value] = rotd50
    if Components.crotd100 in comps_to_store:
        value_dict[Components.crotd100.str_value] = rotd100
    if Components.crotd100_50 in comps_to_store:
        value_dict[Components.crotd100_50.str_value] = rotd100 / rotd50
    return value_dict


def compute_adv_measure(waveform, advanced_im_config, output_dir):
    """
    Wrapper function to call advanced IM workflow
    :param waveform: Tuple of waveform objects (first is acc, second is vel)
    :param advanced_im_config: advanced_im_config Named Tuple containing list of IMs, config file and path to OpenSeeS
    :param output_dir: Directory where output folders are contained. Structure is /path/to/output_dir/station/im_name
    :return:
    """

    waveform_acc = waveform[0]
    station_name = waveform_acc.station_name
    adv_im_out_dir = os.path.join(output_dir, station_name)
    advanced_IM_factory.compute_ims(waveform_acc, advanced_im_config, adv_im_out_dir)


def compute_measure_single(
    waveform, ims, comps_to_store, im_options, comps_to_calculate, progress, logger=get_basic_logger()
):
    """
    Compute measures for a single station
    :param: a tuple consisting 5 params: waveform, ims, comp, period, str_comps
    waveform: a single tuple that contains (waveform_acc,waveform_vel)
    progress: a tuple containing station number and total number of stations
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
    station_i, n_stations = progress
    logger.info(f"Processing {station_name} - {station_i} / {n_stations}")

    result = {(station_name, comp.str_value): {} for comp in comps_to_store}

    im_functions = {
        "PGV": (calc_PG, (velocities,)),
        "PGA": (calc_PG, (accelerations,)),
        "CAV": (calc_CAV, (accelerations, times)),
        "pSA": (
            calculate_pSAs,
            (DT, accelerations, im_options, result, station_name, waveform_acc),
        ),
        "SDI": (
            calculate_SDI,
            (DT, accelerations, im_options, result, station_name, waveform_acc),
        ),
        "FAS": (calc_FAS, (DT, accelerations, im_options, result, station_name)),
        "AI": (calc_AI, (accelerations, times)),
        "SED": (calc_SED, (velocities, times)),
        "MMI": (calc_MMI, (velocities,)),
        "Ds595": (calc_DS, (accelerations, DT, 5, 95)),
        "Ds575": (calc_DS, (accelerations, DT, 5, 75)),
    }

    for im in set(ims).intersection(im_functions.keys()):
        # print(im)
        func, args = im_functions[im]
        values_to_store = func(*args, im, comps_to_store, comps_to_calculate)
        if values_to_store is None:
            # value storing has been handled by the called function
            continue
        for comp in comps_to_store:
            if comp.str_value in values_to_store:
                result[(station_name, comp.str_value)][im] = values_to_store[
                    comp.str_value
                ]

    return result


def sanitise_single_value_arrays(input_dict):
    for key, item in input_dict.items():
        input_dict[key] = np.squeeze(item)


def calc_DS(
    accelerations, dt, perclow, perchigh, im, comps_to_store, comps_to_calculate
):
    value = intensity_measures.getDs_nd(
        accelerations, dt=dt, percLow=perclow, percHigh=perchigh
    )
    values = array_to_dict(value, comps_to_calculate, im, comps_to_store)
    if check_rotd(comps_to_store):
        func = partial(
            intensity_measures.getDs_nd, dt=dt, percLow=perclow, percHigh=perchigh
        )
        rotd = calculate_rotd(accelerations, comps_to_store, func=func)
        values.update(rotd)
    return values


def calc_PG(waveform, im, comps_to_store, comps_to_calculate):
    value = intensity_measures.get_max_nd(waveform)
    values = array_to_dict(value, comps_to_calculate, im, comps_to_store)
    if check_rotd(comps_to_store):
        rotd = calculate_rotd(waveform, comps_to_store)
        sanitise_single_value_arrays(rotd)
        values.update(rotd)
    return values


def calc_CAV(waveform, times, im, comps_to_store, comps_to_calculate):
    value = intensity_measures.get_cumulative_abs_velocity_nd(waveform, times)
    values = array_to_dict(value, comps_to_calculate, im, comps_to_store)
    if check_rotd(comps_to_store):
        func = lambda x: intensity_measures.get_cumulative_abs_velocity_nd(
            np.squeeze(x), times=times
        )
        rotd = calculate_rotd(waveform, comps_to_store, func=func)
        values.update(rotd)
    return values


def calc_MMI(waveform, im, comps_to_store, comps_to_calculate):
    value = intensity_measures.calculate_MMI_nd(waveform)
    values = array_to_dict(value, comps_to_calculate, im, comps_to_store)
    if check_rotd(comps_to_store):
        func = lambda x: intensity_measures.calculate_MMI_nd(np.squeeze(x))
        rotd = calculate_rotd(waveform, comps_to_store, func=func)
        values.update(rotd)
    return values


def calc_SED(velocities, times, im, comps_to_store, comps_to_calculate):
    value = intensity_measures.get_specific_energy_density_nd(velocities, times)
    values = array_to_dict(value, comps_to_calculate, im, comps_to_store)
    if check_rotd(comps_to_store):
        func = lambda x: intensity_measures.get_specific_energy_density_nd(
            np.squeeze(x), times=times
        )
        rotd = calculate_rotd(velocities, comps_to_store, func=func)
        values.update(rotd)
    return values


def calc_AI(accelerations, times, im, comps_to_store, comps_to_calculate):
    value = intensity_measures.get_arias_intensity_nd(accelerations, times)
    values = array_to_dict(value, comps_to_calculate, im, comps_to_store)
    if check_rotd(comps_to_store):
        func = lambda x: intensity_measures.get_arias_intensity_nd(
            np.squeeze(x), times=times
        )
        rotd = calculate_rotd(accelerations, comps_to_store, func=func)
        values.update(rotd)
    return values


def calc_FAS(
    DT,
    accelerations,
    im_options,
    result,
    station_name,
    im,
    comps_to_store,
    comps_to_calculate,
):
    try:
        value = get_fourier_spectrum(accelerations[:, :2], DT, im_options[im])
        values_to_store = array_to_dict(value, comps_to_calculate, im, comps_to_store)
        if check_rotd(comps_to_store):
            func = lambda rotated_waveform: get_fourier_spectrum(
                rotated_waveform.squeeze(), dt=DT, fa_frequencies_int=im_options[im]
            )
            rotd = calculate_rotd(accelerations, comps_to_store, func=func)
            values_to_store.update(rotd)
    except FileNotFoundError as e:
        print(
            f"Attempting to compute fourier spectrum raised exception: {e}\nThis was most likely caused by attempting to compute for a waveform with more than 16384 timesteps."
        )
    else:
        # compute EAS, the euclidean distance of FAS 000 and 090
        if Components.ceas in comps_to_store:
            values_to_store[
                Components.ceas.str_value
            ] = intensity_measures.get_euclidean_dist(value[:, 0], value[:, 1])

        for comp in comps_to_store:
            if comp.str_value in values_to_store:
                for i, val in enumerate(im_options[im]):
                    result[(station_name, comp.str_value)][
                        f"{im}_{str(val)}"
                    ] = values_to_store[comp.str_value][i]


def calculate_pSAs(
    DT,
    accelerations,
    im_options,
    result,
    station_name,
    waveform_acc,
    im,
    comps_to_store,
    comps_to_calculate,
):
    # Get spectral accelerations. Has shape (len(periods), nt)
    spectral_accelerations = intensity_measures.get_spectral_acceleration_nd(
        accelerations, im_options[im], waveform_acc.NT, DT
    )
    # Calculate the maximums of the basic components and pass this to array_to_dict which calculates geom too
    # Store the pSA im values in the format Tuple(List(periods), dict(component: List(im_values)))
    # Where the im_values in the component dictionaries correspond to the periods in the periods list
    pSA_values = array_to_dict(
        np.max(np.abs(spectral_accelerations), axis=1),
        comps_to_calculate,
        im,
        comps_to_store,
    )
    if check_rotd(comps_to_store):
        # Only run if any of the given components are selected (Non empty intersection)
        pSA_values.update(calculate_rotd(spectral_accelerations, comps_to_store))

    for comp in comps_to_store:
        if comp.str_value in pSA_values:
            for i, val in enumerate(im_options[im]):
                result[(station_name, comp.str_value)][f"{im}_{str(val)}"] = pSA_values[
                    comp.str_value
                ][i]


def calculate_SDI(
    DT,
    accelerations,
    im_options,
    result,
    station_name,
    waveform_acc,
    im,
    comps_to_store,
    comps_to_calculate,
    z=0.05,  # damping ratio
    alpha=0.05,  # strain hardening ratio
    dy=0.1765,  # strain hardening ratio
    dt=0.005,  # analysis time step
):
    # Get displacements by Burks_Baker_2013. Has shape (len(periods), nt-1, len(comps))
    displacements = (
        intensity_measures.get_SDI_nd(
            accelerations, im_options[im], waveform_acc.NT, DT, z, alpha, dy, dt
        )
        * 100
    )  # Burks & Baker returns m, but output is stored in cm

    # Calculate the maximums of the basic components and pass this to array_to_dict which calculates geom too
    # Store the SDI im values in the format dict(component: List(im_values))
    # Where the im_values in the component dictionaries correspond to the periods in the periods list
    sdi_values = array_to_dict(
        np.max(np.abs(displacements), axis=1), comps_to_calculate, im, comps_to_store
    )

    if check_rotd(comps_to_store):
        # Only run if any of the given components are selected (Non empty intersection)
        sdi_values.update(calculate_rotd(displacements, comps_to_store))

    for comp in comps_to_store:
        if comp.str_value in sdi_values:
            for i, val in enumerate(im_options[im]):
                result[(station_name, comp.str_value)][f"{im}_{str(val)}"] = sdi_values[
                    comp.str_value
                ][i]


def get_bbseis(input_path, file_type, selected_stations, real_only=False):
    """
    :param input_path: user input path to bb.bin or a folder containing ascii files
    :param file_type: binary or ascii
    :param selected_stations: list of user input stations
    :param real_only: considers real stations only
    :return: bbseries, station_names
    """
    bbseries = None
    if file_type == FILE_TYPE_DICT["b"]:
        bbseries = timeseries.BBSeis(input_path)
        bb_stations = bbseries.stations.name
        if selected_stations is None:
            station_names = bb_stations
        else:
            # making sure selected stations are in bbseis
            station_names = list(set(selected_stations).intersection(bb_stations))
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
    else:
        return
    station_names = list(station_names)
    if real_only:
        station_names = [
            stat_name
            for stat_name in station_names
            if not shared.is_virtual_station(stat_name)
        ]
    assert len(station_names) > 0, "No station is found"
    return bbseries, station_names


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
    advanced_im_config=None,
    real_only=False,
    logger=get_basic_logger()
):
    """
    using multiprocesses to compute measures.
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
    :param units:
    :param advanced_im_config:
    :param real_only:
    :return:
    """
    #  for running adv_im
    running_adv_im = (advanced_im_config is not None) and (
        advanced_im_config.IM_list is not None
    )

    (
        components_to_calculate,
        components_to_store,
    ) = constants.Components.get_comps_to_calc_and_store(comp)

    bbseries, station_names = get_bbseis(
        input_path, file_type, station_names, real_only=real_only
    )
    total_stations = len(station_names)
    # determine the size of each iteration base on num of processes and mem
    steps = get_steps(
        input_path, process, total_stations, "FAS" in ims and bbseries.nt > 32768
    )

    # initialize result list for basic IM
    if not running_adv_im:
        all_results = []

    with Pool(process) as p, ProgressTracker(total_stations) as p_t:
        for i in range(0, total_stations, steps):
            # read waveforms of stations
            # each iteration = steps size
            stations_to_run = station_names[i : i + steps]
            waveforms = read_waveform.read_waveforms(
                input_path,
                bbseries,
                stations_to_run,
                components_to_calculate,
                wave_type=wave_type,
                file_type=file_type,
                units=units,
            )
            # only run basic im if and only if adv_im not going to run
            if running_adv_im:
                adv_array_params = [
                    (waveform, advanced_im_config, output) for waveform in waveforms
                ]
                # calculate IM for stations in this iteration
                p.starmap(compute_adv_measure, adv_array_params)
            else:
                array_params = [
                    (
                        waveform,
                        sorted(ims),
                        sorted(components_to_store, key=lambda x: x.value),
                        im_options,
                        sorted(components_to_calculate, key=lambda x: x.value),
                        (ii, total_stations),
                        logger,
                    )
                    for ii, waveform in enumerate(waveforms, start=i+1)
                ]
                all_results.extend(p.starmap(compute_measure_single, array_params))
            p_t(i)

    if running_adv_im:
        # read, agg and store csv
        advanced_IM_factory.agg_csv(advanced_im_config, station_names, output)
    else:
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


def validate_fas_frequency(arg_fas_freq):
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
    if os.path.isfile(input_path):
        estimated_mem = os.path.getsize(input_path) * MEM_FACTOR
    else:
        # Sum up the total size of the files contained in the folder and use it for estimating the memory use
        estimated_mem = MEM_FACTOR * sum(
            [
                os.path.getsize(os.path.join(input_path, name))
                for name in os.listdir(input_path)
            ]
        )
    if high_mem_usage:
        estimated_mem *= MEM_FACTOR
    available_mem = nps * MEM_PER_CORE
    assert estimated_mem > 0, f"Estimated memory is 0: Check {input_path}"
    batches = np.ceil(np.divide(estimated_mem, available_mem))
    steps = int(np.floor(np.divide(total_stations, batches)))
    if steps == 0:
        steps = total_stations
    return steps
