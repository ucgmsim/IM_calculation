import csv
import getpass
import glob
import os
import sys
from datetime import datetime
from multiprocessing.pool import Pool

import numpy as np

from qcore import timeseries, constants

from IM_calculation.IM import read_waveform, intensity_measures
from IM_calculation.IM.computeFAS import get_fourier_spectrum


G = 981.0
DEFAULT_IMS = ("PGA", "PGV", "CAV", "AI", "Ds575", "Ds595", "MMI", "pSA")
ALL_IMS = ["PGA", "PGV", "CAV", "AI", "Ds575", "Ds595", "MMI", "pSA", "FAS"]


MULTI_VALUE_IMS = ["pSA", "FAS"]

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
FAS_FREQUENCY = np.logspace(-1, 2, num=100, base=10.0)

COMPONENTS = ["090", "000", "ver", "geom", "rotd50", "rotd100", "rotd10050"]

FILE_TYPE_DICT = {"a": "ascii", "b": "binary"}
META_TYPE_DICT = {"s": "simulated", "o": "observed", "u": "unknown"}
RUNNAME_DEFAULT = "all_station_ims"

OUTPUT_PATH = os.path.join("/home", getpass.getuser())
OUTPUT_SUBFOLDER = "stations"

MEM_PER_CORE = 7.5e8
MEM_FACTOR = 4


def convert_str_comp(arg_comps):
    """
    convert arg comps to str_comps_for integer_convestion in read_waveform & str comps for writing result
    :param arg_comps: user input a list of comp(s)
    :return: two lists of str comps
    """
    # comps = ["000", "geom",'ver']
    if "geom" in arg_comps or "rotd50" in arg_comps or "rotd100" in arg_comps or "rotd10050" in arg_comps:
        # ['000', 'ver', '090', 'geom']
        str_comps = list(set(arg_comps).union({"090", "000"}))
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
        if im in MULTI_VALUE_IMS:
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


def compute_measure_single(waveform, ims, comps, im_options, str_comps):
    """
    Compute measures for a single station
    :param: a tuple consisting 5 params: waveform, ims, comp, period, str_comps
    waveform: a single tuple that contains (waveform_acc,waveform_vel)
    :return: {result[station_name]: {[im]: value or (period,value}}
    """
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

    if {"pSA", "rotd50", "rotd100", "rotd100/50"} & set(ims):
        psa, u = intensity_measures.get_spectral_acceleration_nd(
            accelerations, im_options["pSA"], waveform_acc.NT, DT
        )
        if {"SA", "rotd50", "rotd100", "rotd100/50"} & set(ims):
            rotd = intensity_measures.calc_rotd(u[:2])
            rotd50 = sorted(rotd)[len(rotd)//2]
            rotd100 = sorted(rotd)[-1]

    if "PGV" in ims:
        value = intensity_measures.get_max_nd(velocities)
        result[station_name]["PGV"] = array_to_dict(value, str_comps, "PGV", comps)

    if "PGA" in ims:
        value = intensity_measures.get_max_nd(accelerations)
        result[station_name]["PGA"] = array_to_dict(value, str_comps, "PGA", comps)

    if "pSA" in ims:
        # store a im type values into a dict {comp: np_array/single float}
        # Geometric is also calculated here
        result[station_name]["pSA"] = (
            im_options["pSA"],
            array_to_dict(psa, str_comps, "pSA", comps),
        )

    if "FAS" in ims:
        value = get_fourier_spectrum(accelerations, DT, im_options["FAS"])
        result[station_name]["FAS"] = (
            im_options["FAS"],
            (array_to_dict(value, str_comps, "FAS", comps)),
        )

    if "rotd50" in ims:
        value = rotd50
        result[station_name]["rotd50"] = array_to_dict(value, str_comps, "rotd50", comps)

    if "rotd100" in ims:
        value = rotd100
        result[station_name]["rotd100"] = array_to_dict(value, str_comps, "rotd100", comps)

    elif "rotd100/50" in ims:
        value = rotd100 / rotd50
        result[station_name]["rotd100"] = array_to_dict(value, str_comps, "rotd100", comps)

    # TODO: Speed up Ds calculations
    if "Ds595" in ims:
        value = intensity_measures.getDs_nd(DT, accelerations, 5, 95)
        result[station_name]["Ds595"] = array_to_dict(value, str_comps, "Ds595", comps)

    if "Ds575" in ims:
        value = intensity_measures.getDs_nd(DT, accelerations, 5, 75)
        result[station_name]["Ds575"] = array_to_dict(value, str_comps, "Ds575", comps)

    if "AI" in ims:
        value = intensity_measures.get_arias_intensity_nd(accelerations, G, times)
        result[station_name]["AI"] = array_to_dict(value, str_comps, "AI", comps)

    if "CAV" in ims:
        value = intensity_measures.get_cumulative_abs_velocity_nd(accelerations, times)
        result[station_name]["CAV"] = array_to_dict(value, str_comps, "CAV", comps)

    if "MMI" in ims:
        value = intensity_measures.calculate_MMI_nd(velocities)
        result[station_name]["MMI"] = array_to_dict(value, str_comps, "MMI", comps)

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
    str_comps_for_int, str_comps = convert_str_comp(comp)

    bbseries, station_names = get_bbseis(input_path, file_type, station_names)

    total_stations = len(station_names)
    steps = get_steps(input_path, process, total_stations)

    all_result_dict = {}
    p = Pool(process)

    i = 0
    while i < total_stations:
        waveforms = read_waveform.read_waveforms(
            input_path,
            bbseries,
            station_names[i : i + steps],
            str_comps_for_int,
            wave_type=wave_type,
            file_type=file_type,
            units=units,
        )
        i += steps
        array_params = []
        for waveform in waveforms:
            array_params.append((waveform, ims, comp, im_options, str_comps))

        result_list = p.starmap(compute_measure_single, array_params)

        for result in result_list:
            all_result_dict.update(result)

    write_result(
        all_result_dict, output, identifier, comp, ims, im_options, simple_output
    )

    generate_metadata(output, identifier, rupture, run_type, version)


def get_result_filepath(output_folder, arg_identifier, suffix):
    return os.path.join(output_folder, "{}{}".format(arg_identifier, suffix))


def get_header(ims, im_options):
    """
    write header colums for output im_calculations csv file
    :param ims: a list of im measures
    :param im_options: a list of pSA periods
    :return: header
    """
    header = ["station", "component"]
    psa_names = []

    for im in ims:
        if im in MULTI_VALUE_IMS:  # only write period if im is pSA.
            for p in im_options[im]:
                if p in BSC_PERIOD:
                    psa_names.append(f"{im}_{p}")
                else:
                    psa_names.append("{}_{:.12f}".format(im, p))
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
            if im not in MULTI_VALUE_IMS:
                row.append(result_dict[station][im][c])
            else:
                row += result_dict[station][im][1][c].tolist()
        big_csv_writer.writerow(row)
        if sub_csv_writer:
            sub_csv_writer.writerow(row)


def write_result(
    result_dict, output_folder, identifier, comps, ims, im_options, simple_output
):
    """
    write a big csv that contains all calculated im value and single station csvs
    :param result_dict:
    :param output_folder:
    :param identifier: user input run name
    :param comps: a list of comp(s)
    :param ims: a list of im(s)
    :param im_options:
    :param simple_output
    :return:output result into csvs
    """
    output_path = get_result_filepath(output_folder, identifier, ".csv")
    header = get_header(ims, im_options)

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


def get_im_or_period_help(available_values, default_values, im_or_period):
    """
    :param available_values: All available predefined constants. Must be a superset of default_values
    :param default_values: predefined constants
    :param im_or_period: should be either string "im" or string "period"
    :return: a help message for input component arg
    """
    return "Available {}s are: {} and default {}s are: {}".format(
        im_or_period,
        ",".join(str(v) for v in available_values),
        im_or_period,
        ",".join(str(v) for v in default_values),
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


def validate_im(parser, arg_im, components):
    """
    returns validated user input if pass the validation else raise parser error
    :param parser:
    :param arg_im: input
    :return: validated im(s) in a list
    """
    im = arg_im
    if im != ALL_IMS:
        for m in im:
            if m not in ALL_IMS:
                parser.error(
                    "please enter valid im measure name. {}".format(
                        get_im_or_period_help(ALL_IMS, DEFAULT_IMS, "IM")
                    )
                )
    return im


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
        period = np.unique(np.append(period, EXT_PERIOD))

    return period


def validate_FAS_frequency(arg_fas_freq):
    """
    returns validated user input if pass the validation else raise parser error
    :param arg_fas_freq: input
    :return: frequencies in a numpy array
    """
    frequencies = np.array(arg_fas_freq, dtype="float64")

    return frequencies


def get_steps(input_path, nps, total_stations):
    """
    :param input_path: user input file/dir path
    :param nps: number of processes
    :param total_stations: total number of stations
    :return: number of stations per iteration/batch
    """
    estimated_mem = os.stat(input_path).st_size * MEM_FACTOR
    available_mem = nps * MEM_PER_CORE
    batches = np.ceil(np.divide(estimated_mem, available_mem))
    steps = int(np.floor(np.divide(total_stations, batches)))
    if steps == 0:
        steps = total_stations
    return steps
