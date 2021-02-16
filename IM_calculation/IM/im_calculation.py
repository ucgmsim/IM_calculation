import csv
import glob
import os
import re
import sys

from datetime import datetime
from functools import partial
from multiprocessing.pool import Pool
from collections import ChainMap
from typing import List, Iterable

import numpy as np
import pandas as pd

from qcore import timeseries, constants
from qcore.constants import Components
from qcore.im import order_im_cols_df

from IM_calculation.Advanced_IM import advanced_IM_factory
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


def add_if_not_exist(component_list, component):
    if not component in component_list:
        component_list.append(component)
    return component_list


def convert_str_comp(arg_comps):
    """
    convert arg comps to str_comps_for integer_convestion in read_waveform & str comps for writing result
    :param arg_comps: user input a list of comp(s)
    :return: two lists of str comps
    """
    # comps = ["000", "geom",'ver']
    if "geom" in arg_comps:
        # ['000', 'ver', '090', 'geom']
        str_comps = add_if_not_exist(arg_comps, "090")
        str_comps = add_if_not_exist(str_comps, "000")
        # for integer convention, str_comps shouldn't include geom as waveform has max 3 components
        str_comps.remove("geom")
        # for writing result, make a copy of the str_comps for int convention, and shift geom to the end
        str_comps_for_writing = str_comps[:]
        str_comps_for_writing.append("geom")
        return str_comps, str_comps_for_writing
    else:
        return arg_comps, arg_comps


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
        {
            Components.crotd50,
            Components.crotd100,
            Components.crotd100_50,
        }.intersection(comps_to_store)
    )


def calculate_rotd(
    spectral_displacements,
    comps_to_store: List[Components],
    func=lambda x: np.max(np.abs(x), axis=1),
):
    """
    Calculates rotd for given spectral displacements
    :param spectral_displacements: An array with shape [periods.size, nt, 2] where nt is the number of timesteps in the original waveform
    :param comps_to_store: A list of components to store
    :param func: The function to apply to the rotated waveforms. Defaults to taking the maximum absolute value across all rotations (used by PGA, PGV, pSA)
    :return: A dictionary with the comps_to_store as keys, and 1d arrays of shape [periods.size] containing the rotd values
    """
    rotd = func(intensity_measures.get_rotations(spectral_displacements))
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
    try:
        if advanced_im_config.IM_list is not None:
            waveform_acc = waveform[0]
            station_name = waveform_acc.station_name
            adv_im_out_dir = os.path.join(output_dir, station_name)
            advanced_IM_factory.compute_ims(
                waveform_acc, advanced_im_config, adv_im_out_dir
            )
    except AttributeError:
        print(
            "cannot access IM_list under advanced_im_config : {}".format(
                advanced_im_config
            )
        )


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

    im_functions = {
        "PGV": (calc_PG, (velocities,)),
        "PGA": (calc_PG, (accelerations,)),
        "CAV": (calc_CAV, (accelerations, times)),
        "pSA": (
            calculate_pSAs,
            (DT, accelerations, im_options, result, station_name, waveform_acc),
        ),
        "FAS": (calc_FAS, (DT, accelerations, im_options, result, station_name)),
        "AI": (calc_AI, (accelerations, G, times)),
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
            intensity_measures.getDs_nd,
            dt=dt,
            percLow=perclow,
            percHigh=perchigh,
        )
        rotd = calculate_rotd(
            np.expand_dims(accelerations, 0), comps_to_store, func=func
        )
        values.update(rotd)
    return values


def calc_PG(waveform, im, comps_to_store, comps_to_calculate):
    value = intensity_measures.get_max_nd(waveform)
    values = array_to_dict(value, comps_to_calculate, im, comps_to_store)
    if check_rotd(comps_to_store):
        rotd = calculate_rotd(np.expand_dims(waveform, 0), comps_to_store)
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
        rotd = calculate_rotd(np.expand_dims(waveform, 0), comps_to_store, func)
        values.update(rotd)
    return values


def calc_MMI(waveform, im, comps_to_store, comps_to_calculate):
    value = intensity_measures.calculate_MMI_nd(waveform)
    values = array_to_dict(value, comps_to_calculate, im, comps_to_store)
    if check_rotd(comps_to_store):
        func = lambda x: intensity_measures.calculate_MMI_nd(np.squeeze(x))
        rotd = calculate_rotd(np.expand_dims(waveform, 0), comps_to_store, func)
        values.update(rotd)
    return values


def calc_AI(accelerations, G, times, im, comps_to_store, comps_to_calculate):
    value = intensity_measures.get_arias_intensity_nd(accelerations, G, times)
    values = array_to_dict(value, comps_to_calculate, im, comps_to_store)
    if check_rotd(comps_to_store):
        func = lambda x: intensity_measures.get_arias_intensity_nd(
            np.squeeze(x), g=G, times=times
        )
        rotd = calculate_rotd(
            np.expand_dims(accelerations, 0), comps_to_store, func=func
        )
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
            rotd = calculate_rotd(
                np.expand_dims(accelerations, 0), comps_to_store, func=func
            )
            values_to_store.update(rotd)
    except FileNotFoundError as e:
        print(
            f"Attempting to compute fourier spectrum raised exception: {e}\nThis was most likely caused by attempting to compute for a waveform with more than 16384 timesteps."
        )
    else:
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
    # store a im type values into a dict {comp: np_array/single float}
    # Geometric is also calculated here
    psa, spectral_displacements = intensity_measures.get_spectral_acceleration_nd(
        accelerations, im_options[im], waveform_acc.NT, DT
    )
    # Store the pSA im values in the format Tuple(List(periods), dict(component: List(im_values)))
    # Where the im_values in the component dictionaries correspond to the periods in the periods list
    pSA_values = array_to_dict(psa, comps_to_calculate, im, comps_to_store)

    if check_rotd(comps_to_store):
        # Only run if any of the given components are selected (Non empty intersection)
        rotd = calculate_rotd(spectral_displacements, comps_to_store)
        pSA_values.update(rotd)

    for comp in comps_to_store:
        if comp.str_value in pSA_values:
            for i, val in enumerate(im_options[im]):
                result[(station_name, comp.str_value)][f"{im}_{str(val)}"] = pSA_values[
                    comp.str_value
                ][i]


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
    else:
        return
    return bbseries, list(station_names)


def agg_csv(stations, im_calc_dir, im_type):
    # get csv base on station name
    # quick check of args format
    if type(im_type) != str:
        raise TypeError(
            "im_type should be a string, but get {} instead".format(type(im_type))
        )
    # initial a blank dataframe
    df = pd.DataFrame()

    # loop through all stations
    for station in stations:
        # use glob(?) and qcore.sim_struc to get specific station_im.csv
        # TODO: define this structure into qcore.sim_struct
        station_im_dir = os.path.join(im_calc_dir, station)
        im_type_path = os.path.join(station_im_dir, im_type)
        im_path = os.path.join(im_type_path, im_type + ".csv")
        # read a df and add station name as colum
        df_tmp = pd.read_csv(im_path)

        # add in the station name before agg
        df_tmp.insert(0, "station", station)

        # append the df
        df = df.append(df_tmp)

    # leave write csv to parent function
    return df


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
    advanced_im_config=advanced_IM_factory.advanced_im_config(None, None, None),
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
    if advanced_im_config.IM_list:
        df_adv_im = {im: pd.DataFrame() for im in advanced_im_config.IM_list}

    i = 0
    while i < total_stations:
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
        i += steps
        array_params = []
        adv_array_params = []
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
            adv_array_params.append((waveform, advanced_im_config, output_dir))
        # only run simply im if and only if adv_im not going to run
        if not advanced_im_config.IM_list:
            all_results.extend(p.starmap(compute_measure_single, array_params))
            # adv_im changes for ref, TODO:del once test pass
            # result_list = p.map(compute_measure_single, array_params)
            # for result in result_list:
            #    all_result_dict.update(result)
        if advanced_im_config.IM_list:
            # calculate IM for stations in this iteration
            p.starmap(compute_adv_measure, adv_array_params)
            # read and agg data into a pandas array
            # loop through all im_type in advanced_im_config
            for im_type in advanced_im_config.IM_list:
                # agg_csv(stations_to_run, output_dir, im_type)
                df_adv_im[im_type] = df_adv_im[im_type].append(
                    agg_csv(stations_to_run, output_dir, im_type)
                )

    # write the ouput after all cals are done
    if not advanced_im_config.IM_list:
        # TODO: del after test pass
        # write_result(
        #    all_result_dict, output_dir, identifier, comp, ims, period, simple_output
        # )

        all_result_dict = ChainMap(*all_results)
        output_path = get_result_filepath(output, identifier, ".csv")

        results_dataframe = pd.DataFrame.from_dict(all_result_dict, orient="index")
        results_dataframe.index = pd.MultiIndex.from_tuples(
            results_dataframe.index, names=["station", "component"]
        )
        results_dataframe.sort_values(["station", "component"], inplace=True)
        results_dataframe = order_im_cols_df(results_dataframe)

        # Save the transposed dataframe
        results_dataframe.to_csv(output_path)

    # write for advanced IM (pandas array)
    if advanced_im_config.IM_list:
        # dump the whole array
        for im_type in advanced_im_config.IM_list:
            # do a natural sort on the column names
            df_adv_im[im_type] = df_adv_im[im_type][
                list(df_adv_im[im_type].columns[:2])
                + sorted(df_adv_im[im_type].columns[2:], key=natural_key)
            ]
            # check if file exist already, if exist header=False
            adv_im_out = os.path.join(output, im_type + ".csv")
            print("Dumping adv_im data to : {}".format(adv_im_out))
            if os.path.isfile(adv_im_out):
                print_header = False
            else:
                print_header = True
            df_adv_im[im_type].to_csv(
                adv_im_out, mode="a", header=print_header, index=False
            )
    generate_metadata(output, identifier, rupture, run_type, version)


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)]


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
    estimated_mem = os.stat(input_path).st_size * MEM_FACTOR
    if high_mem_usage:
        estimated_mem *= MEM_FACTOR
    available_mem = nps * MEM_PER_CORE
    batches = np.ceil(np.divide(estimated_mem, available_mem))
    steps = int(np.floor(np.divide(total_stations, batches)))
    if steps == 0:
        steps = total_stations
    return steps
