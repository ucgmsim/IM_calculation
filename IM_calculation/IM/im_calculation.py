import csv
from datetime import datetime
import getpass
import glob
import os
import re
import sys

import numpy as np
from multiprocessing.pool import Pool
import pandas as pd

from IM_calculation.Advanced_IM import advanced_IM_factory
from IM_calculation.IM import read_waveform
from IM_calculation.IM import intensity_measures
from qcore import constants
from qcore import timeseries


G = 981.0
IMS = ["PGA", "PGV", "CAV", "AI", "Ds575", "Ds595", "MMI", "pSA"]

EXT_PERIOD = np.logspace(start=np.log10(0.01), stop=np.log10(10.0), num=100, base=10)
BSC_PERIOD = [
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    7.5,
    10.0,
]

COMPONENTS = ["090", "000", "ver", "geom"]

FILE_TYPE_DICT = {"a": "ascii", "b": "binary"}
META_TYPE_DICT = {"s": "simulated", "o": "observed", "u": "unknown"}
RUNNAME_DEFAULT = "all_station_ims"

OUTPUT_PATH = os.path.join("/home", getpass.getuser())
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


def array_to_dict(value, str_comps, im, arg_comps):
    """
    convert a numpy arrary that contains calculated im values to a dict {comp: value}
    :param value: calculated intensity measure for a waveform
    :param str_comps: a list of components converted from user input
    :param im:
    :param arg_comps:user input list of components
    :return: a dict {comp: value}
    """
    value_dict = {}
    # ["090", "ver"], ["090", "000", "geom"], ["090", "000", "ver", "geom"]
    # [0, 2]          [0, 1]                  [0, 1, 2]
    for i in range(value.shape[-1]):
        # pSA returns a 2D array
        if im == "pSA":
            value_dict[str_comps[i]] = value[:, i]
        else:
            value_dict[str_comps[i]] = value[i]
    # In this case, if geom in str_comps,
    # it's guaranteed that 090 and 000 will be present in value_dict
    if "geom" in str_comps:
        value_dict["geom"] = intensity_measures.get_geom(
            value_dict["090"], value_dict["000"]
        )
        # then we pop unwanted keys from value_dict
        for k in str_comps:
            if k not in arg_comps:
                del value_dict[k]
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


def compute_measure_single(value_tuple):
    """
    Compute measures for a single station
    :param: a tuple consisting 5 params: waveform, ims, comp, period, str_comps
    waveform: a single tuple that contains (waveform_acc,waveform_vel)
    :return: {result[station_name]: {[im]: value or (period,value}}
    """
    waveform, ims, comps, period, str_comps = value_tuple
    result = {}
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

    result[station_name] = {}

    for im in ims:
        if im == "PGV":
            value = intensity_measures.get_max_nd(velocities)

        if im == "PGA":
            value = intensity_measures.get_max_nd(accelerations)

        if im == "pSA":
            value = intensity_measures.get_spectral_acceleration_nd(
                accelerations, period, waveform_acc.NT, DT
            )

        # TODO: Speed up Ds calculations
        if im == "Ds595":
            value = intensity_measures.getDs_nd(DT, accelerations, 5, 95)

        if im == "Ds575":
            value = intensity_measures.getDs_nd(DT, accelerations, 5, 75)

        if im == "AI":
            value = intensity_measures.get_arias_intensity_nd(accelerations, G, times)

        if im == "CAV":
            value = intensity_measures.get_cumulative_abs_velocity_nd(
                accelerations, times
            )

        if im == "MMI":
            value = intensity_measures.calculate_MMI_nd(velocities)

        # store a im type values into a dict {comp: np_array/single float}
        # Geometric is also calculated here
        value_dict = array_to_dict(value, str_comps, im, comps)

        # store value dict into the biggest result dict
        if im == "pSA":
            result[station_name][im] = (period, value_dict)
        else:
            result[station_name][im] = value_dict

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
        bb_stations = bbseries.stations.name
        if selected_stations is None:
            station_names = bb_stations 
        else:
            station_list_tmp = []
            for staion_name in selected_stations:
                if staion_name in bb_stations:
                    station_list_tmp.append(staion_name)
            station_names = station_list_tmp
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
    ims=IMS,
    comp=None,
    period=None,
    output_dir=None,
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
    :param period:
    :param output_dir:
    :param identifier:
    :param rupture:
    :param run_type:
    :param version:
    :param process:
    :param simple_output:
    :return:
    """
    str_comps_for_int, str_comps = convert_str_comp(comp)
    #print(station_names)
    bbseries, station_names = get_bbseis(input_path, file_type, station_names)
    #print(station_names)
    total_stations = len(station_names)
    # determine the size of each iteration base on num of processers
    steps = get_steps(input_path, process, total_stations)

    all_result_dict = {}
    p = Pool(process)
    if advanced_im_config.IM_list:
        df_adv_im = {im: pd.DataFrame() for im in advanced_im_config.IM_list}

    i = 0
    while i < total_stations:
        # read waveforms of stations
        # each iteration = steps size
        stations_to_run = station_names[i : i + steps]
        #print(f"stations_to_run {stations_to_run} {i} steps: {steps}")
        waveforms = read_waveform.read_waveforms(
            input_path,
            bbseries,
            stations_to_run,
            str_comps_for_int,
            wave_type=wave_type,
            file_type=file_type,
            units=units,
        )
        i += steps
        array_params = []
        adv_array_params = []
        for waveform in waveforms:
            array_params.append((waveform, ims, comp, period, str_comps))
            adv_array_params.append((waveform, advanced_im_config, output_dir))
        # only run simply im if and only if adv_im not going to run
        if not advanced_im_config.IM_list:
            result_list = p.map(compute_measure_single, array_params)
            for result in result_list:
                all_result_dict.update(result)
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
        write_result(
            all_result_dict, output_dir, identifier, comp, ims, period, simple_output
        )
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
            adv_im_out = os.path.join(output_dir, im_type + ".csv")
            print("Dumping adv_im data to : {}".format(adv_im_out))
            if os.path.isfile(adv_im_out):
                print_header = False
            else:
                print_header = True
            df_adv_im[im_type].to_csv(
                adv_im_out, mode="a", header=print_header, index=False
            )
    generate_metadata(output_dir, identifier, rupture, run_type, version)


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)]


def get_result_filepath(output_folder, arg_identifier, suffix):
    return os.path.join(output_folder, "{}{}".format(arg_identifier, suffix))


def get_header(ims, period):
    """
    write header colums for output im_calculations csv file
    :param ims: a list of im measures
    :param period: a list of pSA periods
    :return: header
    """
    header = ["station", "component"]
    psa_names = []

    for im in ims:
        if im == "pSA":  # only write period if im is pSA.
            for p in period:
                if p in BSC_PERIOD:
                    psa_names.append("pSA_{}".format(p))
                else:
                    psa_names.append("pSA_{:.12f}".format(p))
            header += psa_names
        else:
            header.append(im)
    return header


def write_rows(comps, station, ims, result_dict, big_csv_writer, sub_csv_writer=None):
    """
    write rows to big csv and, also to single station csvs if not simple output
    :param comps:
    :param station:
    :param ims:
    :param result_dict:
    :param big_csv_writer:
    :param sub_csv_writer:
    :return: write a single row
    """
    for c in comps:
        row = [station, c]
        for im in ims:
            if im != "pSA":
                row.append(result_dict[station][im][c])
            else:
                row += result_dict[station][im][1][c].tolist()
        big_csv_writer.writerow(row)
        if sub_csv_writer:
            sub_csv_writer.writerow(row)


def write_result(
    result_dict, output_folder, identifier, comps, ims, period, simple_output
):
    """
    write a big csv that contains all calculated im value and single station csvs
    :param result_dict:
    :param output_folder:
    :param identifier: user input run name
    :param comps: a list of comp(s)
    :param ims: a list of im(s)
    :param period:
    :param simple_output
    :return:output result into csvs
    """
    output_path = get_result_filepath(output_folder, identifier, ".csv")
    header = get_header(ims, period)

    # big csv containing all stations
    with open(output_path, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",", quotechar="|")
        csv_writer.writerow(header)
        stations = sorted(result_dict.keys())

        # write each row
        for station in stations:
            if not simple_output:  # if single station csvs are needed
                station_csv = os.path.join(
                    output_folder,
                    OUTPUT_SUBFOLDER,
                    "{}_{}.csv".format(station, "_".join(comps)),
                )
                with open(station_csv, "w") as sub_csv_file:
                    sub_csv_writer = csv.writer(
                        sub_csv_file, delimiter=",", quotechar="|"
                    )
                    sub_csv_writer.writerow(header)
                    write_rows(
                        comps,
                        station,
                        ims,
                        result_dict,
                        csv_writer,
                        sub_csv_writer=sub_csv_writer,
                    )
            else:  # if only the big summary csv is needed
                write_rows(comps, station, ims, result_dict, csv_writer)


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


def get_im_or_period_help(default_values, im_or_period):
    """
    :param default_values: predefined constants
    :param im_or_period: should be either string "im" or string "period"
    :return: a help message for input component arg
    """
    return "Available and default {}s are: {}".format(
        im_or_period, ",".join(str(v) for v in default_values)
    )


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


def validate_im(parser, arg_im):
    """
    returns validated user input if pass the validation else raise parser error
    :param parser:
    :param arg_im: input
    :return: validated im(s) in a list
    """
    im = arg_im
    if im != IMS:
        for m in im:
            if m not in IMS:
                parser.error(
                    "please enter valid im measure name. {}".format(
                        get_im_or_period_help(IMS, "IM")
                    )
                )
    return im


def validate_period(parser, arg_period, arg_extended_period, im):
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
        period = np.unique(np.append(period, EXT_PERIOD))

    if (arg_extended_period or period.any()) and "pSA" not in im:
        parser.error(
            "period or extended period must be used with pSA, but pSA is not in the "
            "IM measures entered"
        )

    return period


def get_steps(input_path, nps, total_stations):
    """
    :param input_path: user input file/dir path
    :param nps: number of processes
    :param total_stations: total number of stations
    :return: number of stations per iteration/batch
    """
    estimated_mem = os.stat(input_path).st_size * MEM_FACTOR
    #print(f"estimated_mem {estimated_mem}")
    available_mem = nps * MEM_PER_CORE
    #print(f" available_mem {available_mem}")
    #print(f"total_stations {total_stations}")
    batches = np.ceil(np.divide(estimated_mem, available_mem))
    #print(f"batches f{batches}")
    steps = int(np.floor(np.divide(total_stations, batches)))
    if steps == 0:
        steps = total_stations
    return steps
