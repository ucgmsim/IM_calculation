# TODO when only geom is needed, only 090 and 000 should be calculated
"""
Calculate im values.
Output computed measures to /home/$user/computed_measures if no output path is specified
command:
   python calculate_ims.py test/test_calculate_ims/sample1/input/single_files/ a
   python calculate_ims.py ../BB.bin b
   python calculate_ims.py ../BB.bin b -o /home/yzh231/ -i Albury_666_999 -r Albury -t s -v 18p3 -n 112A CMZ -m PGV pSA -p 0.02 0.03 -e -c geom -np 2
"""

import os
import csv
import argparse
import getpass
import glob
import numpy as np
import sys
from collections import OrderedDict
from datetime import datetime
from IM import intensity_measures
from IM import read_waveform
from qcore import utils, timeseries, pool_wrapper
import pickle

test_data_save_dir = '/home/jpa198/test_space/im_calc_test/pickled/'
REALISATION = 'PangopangoF29_HYP01-10_S1244'
data_taken = {'convert_str_comp': False,
              'array_to_dict': False,
              'compute_measure_single': False,
              'get_bbseis': False,
              'compute_measures_multiprocess': False,
              'get_result_filepath': False,
              'get_header': False,
              'get_comp_name_and_list': False,
              'write_rows': False,
              'write_result': False,
              'generate_metadata': False,
              'get_comp_help': False,
              'get_im_or_period_help': False,
              'validate_input_path': False,
              'validate_comp': False,
              'validate_im': False,
              'validate_period': False,
              'get_steps': False,
              }


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

IDX_EXT_DICT = OrderedDict([(0, "090"), (1, "000"), (2, "ver"), (3, "geom")])
EXT_IDX_DICT = OrderedDict((v, k) for k, v in IDX_EXT_DICT.items())

FILE_TYPE_DICT = {"a": "ascii", "b": "binary"}
META_TYPE_DICT = {"s": "simulated", "o": "observed", "u": "unknown"}

OUTPUT_PATH = os.path.join("/home", getpass.getuser())
OUTPUT_SUBFOLDER = "stations"

RUNNAME_DEFAULT = "all_station_ims"

MEM_PER_CORE = 7.5e8
MEM_FACTOR = 4


def convert_str_comp(comp):
    """
    convert string comp eg '090'/'ellipsis' to int 0/Ellipsis obj
    :param comp: user input
    :return: converted comp
    """
    function = 'convert_str_comp'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_comp.P'), 'wb') as save_file:
            pickle.dump(comp, save_file)

    if comp == "ellipsis":
        converted_comp = Ellipsis
    else:
        converted_comp = EXT_IDX_DICT[comp]

    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_converted_comp.P'), 'wb') as save_file:
            pickle.dump(converted_comp, save_file)
        data_taken[function] = True
    return converted_comp


def array_to_dict(value, comp, converted_comp, im):
    """
    convert a numpy arrary that contains calculated im values to a dict {comp: value}
    :param value:
    :param comp:
    :param converted_comp:
    :param im:
    :return: a dict {comp: value}
    """
    function = 'array_to_dict'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_value.P'), 'wb') as save_file:
            pickle.dump(value, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_comp.P'), 'wb') as save_file:
            pickle.dump(comp, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_converted_comp.P'), 'wb') as save_file:
            pickle.dump(converted_comp, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_im.P'), 'wb') as save_file:
            pickle.dump(im, save_file)

    value_dict = {}
    if converted_comp == Ellipsis:
        comps = list(EXT_IDX_DICT.keys())
        for c in comps[:-1]:  # excludes geom
            column = EXT_IDX_DICT[c]
            if im == "pSA":  # pSA returns 2d array
                d1 = value[:, 0]
                d2 = value[:, 1]
                value_dict[c] = value[:, column]
            else:
                d1 = value[0]
                d2 = value[1]
                value_dict[c] = value[column]
            geom_value = intensity_measures.get_geom(d1, d2)
        value_dict[comps[-1]] = geom_value  # now set geom
    else:
        # only one comp
        # geom_value = None
        if im == "MMI":
            value = value.item(0)  # mmi somehow returns a single array instead of a num
        value_dict[comp] = value

    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_value_dict.P'), 'wb') as save_file:
            pickle.dump(value_dict, save_file)
        data_taken[function] = True

    return value_dict


def compute_measure_single(value_tuple):
    """
    Compute measures for a single station
    :param: a tuple consisting 4 params: waveform, ims, comp, period
    waveform: a single tuple that contains (waveform_acc,waveform_vel)
    :return: {result[station_name]: {[im]: value or (period,value}}
    """

    function = 'compute_measure_single'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_value_tuple.P'), 'wb') as save_file:
            pickle.dump(value_tuple, save_file)

    waveform, ims, comp, period = value_tuple
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
    converted_comp = convert_str_comp(comp)

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
        value_dict = array_to_dict(value, comp, converted_comp, im)

        # store value dict into the biggest result dict
        if im == "pSA":
            result[station_name][im] = (period, value_dict)
        else:
            result[station_name][im] = value_dict

    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_result.P'), 'wb') as save_file:
            pickle.dump(result, save_file)
        data_taken[function] = True

    return result


def get_bbseis(input_path, file_type, selected_stations):
    """
    :param input_path: user input path to bb.bin or a folder containing ascii files
    :param file_type: binary or ascii
    :param selected_stations: list of user input stations
    :return: bbseries, station_names
    """
    function = 'get_bbseis'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_input_path.P'), 'wb') as save_file:
            pickle.dump(input_path, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_file_type.P'), 'wb') as save_file:
            pickle.dump(file_type, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_selected_stations.P'), 'wb') as save_file:
            pickle.dump(selected_stations, save_file)

    bbseries = None
    if file_type == FILE_TYPE_DICT["b"]:
        bbseries = timeseries.BBSeis(input_path)
        if selected_stations is None:
            station_names = bbseries.stations.name
        else:
            station_names = selected_stations

    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_bbseries.P'), 'wb') as save_file:
            pickle.dump(bbseries, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_station_names.P'), 'wb') as save_file:
            pickle.dump(list(station_names), save_file)
        data_taken[function] = True
    return bbseries, list(station_names)


def compute_measures_multiprocess(
    input_path,
    file_type,
    geom_only,
    wave_type,
    station_names,
    ims=IMS,
    comp=None,
    period=None,
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
    :param geom_only:
    :param wave_type:
    :param station_names:
    :param ims:
    :param comp:
    :param period:
    :param output:
    :param identifier:
    :param rupture:
    :param run_type:
    :param version:
    :param process:
    :param simple_output:
    :return:
    """
    function = 'compute_measures_multiprocess'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_input_path.P'), 'wb') as save_file:
            pickle.dump(input_path, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_file_type.P'), 'wb') as save_file:
            pickle.dump(file_type, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_geom_only.P'), 'wb') as save_file:
            pickle.dump(geom_only, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_wave_type.P'), 'wb') as save_file:
            pickle.dump(wave_type, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_station_names.P'), 'wb') as save_file:
            pickle.dump(station_names, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_ims.P'), 'wb') as save_file:
            pickle.dump(ims, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_comp.P'), 'wb') as save_file:
            pickle.dump(comp, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_period.P'), 'wb') as save_file:
            pickle.dump(period, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_output.P'), 'wb') as save_file:
            pickle.dump(output, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_identifier.P'), 'wb') as save_file:
            pickle.dump(identifier, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_rupture.P'), 'wb') as save_file:
            pickle.dump(rupture, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_run_type.P'), 'wb') as save_file:
            pickle.dump(run_type, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_version.P'), 'wb') as save_file:
            pickle.dump(version, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_process.P'), 'wb') as save_file:
            pickle.dump(process, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_simple_output.P'), 'wb') as save_file:
            pickle.dump(simple_output, save_file)
        data_taken[function] = True

    converted_comp = convert_str_comp(comp)

    bbseries, station_names = get_bbseis(input_path, file_type, station_names)

    total_stations = len(station_names)
    steps = get_steps(input_path, process, total_stations)

    all_result_dict = {}
    p = pool_wrapper.PoolWrapper(process)

    i = 0
    while i < total_stations:
        waveforms = read_waveform.read_waveforms(
            input_path,
            bbseries,
            station_names[i : i + steps],
            converted_comp,
            wave_type=wave_type,
            file_type=file_type,
            units=units,
        )
        i += steps
        array_params = []
        for waveform in waveforms:
            array_params.append((waveform, ims, comp, period))

        result_list = p.map(compute_measure_single, array_params)

        for result in result_list:
            all_result_dict.update(result)

    write_result(
        all_result_dict, output, identifier, comp, ims, period, geom_only, simple_output
    )

    generate_metadata(output, identifier, rupture, run_type, version)


def get_result_filepath(output_folder, arg_identifier, suffix):
    function = 'get_result_filepath'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_output_folder.P'), 'wb') as save_file:
            pickle.dump(output_folder, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_arg_identifier.P'), 'wb') as save_file:
            pickle.dump(arg_identifier, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_suffix.P'), 'wb') as save_file:
            pickle.dump(suffix, save_file)
    ret_val = os.path.join(output_folder, "{}{}".format(arg_identifier, suffix))
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_ret_val.P'), 'wb') as save_file:
            pickle.dump(ret_val, save_file)
        data_taken[function] = True
    return ret_val


def get_header(ims, period):
    """
    write header colums for output im_calculations csv file
    :param ims: a list of im measures
    :param period: a list of pSA periods
    :return: header
    """
    function = 'get_header'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_ims.P'), 'wb') as save_file:
            pickle.dump(ims, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_period.P'), 'wb') as save_file:
            pickle.dump(period, save_file)
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
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_header.P'), 'wb') as save_file:
            pickle.dump(header, save_file)
        data_taken[function] = True
    return header


def get_comp_name_and_list(comp, geom_only):
    """
    get comp_name to become part of the sub station csv name
    get comp list for witting rows to big and sub station csvs
    :param station:
    :param comp: a string comp, eg'090'
    :param geom_only: boolean
    :param output_folder:
    :return: comp_list, geom_only flag
    """
    function = 'get_comp_name_and_list'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_comp.P'), 'wb') as save_file:
            pickle.dump(comp, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_geom_only.P'), 'wb') as save_file:
            pickle.dump(geom_only, save_file)
    if geom_only:
        comp_name = "_geom"
        comps = ["geom"]

    elif comp == "ellipsis":
        comp_name = ""
        comps = list(EXT_IDX_DICT.keys())

    else:
        comp_name = "_{}".format(comp)
        comps = [comp]

    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_comp_name.P'), 'wb') as save_file:
            pickle.dump(comp_name, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_comps.P'), 'wb') as save_file:
            pickle.dump(comps, save_file)
        data_taken[function] = True
    return comp_name, comps


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
    function = 'write_rows'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_comps.P'), 'wb') as save_file:
            pickle.dump(comps, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_station.P'), 'wb') as save_file:
            pickle.dump(station, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_ims.P'), 'wb') as save_file:
            pickle.dump(ims, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_result_dict.P'), 'wb') as save_file:
            pickle.dump(result_dict, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_big_csv_writer.P'), 'wb') as save_file:
            pickle.dump(big_csv_writer, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_sub_csv_writer.P'), 'wb') as save_file:
            pickle.dump(sub_csv_writer, save_file)
        data_taken[function] = True
    for c in comps:
        row = [station.decode(), c]
        for im in ims:
            if im != "pSA":
                row.append(result_dict[station][im][c])
            else:
                row += result_dict[station][im][1][c].tolist()
        big_csv_writer.writerow(row)
        if sub_csv_writer:
            sub_csv_writer.writerow(row)


def write_result(
    result_dict, output_folder, identifier, comp, ims, period, geom_only, simple_output
):
    """
    write a big csv that contains all calculated im value and single station csvs
    :param result_dict:
    :param output_folder:
    :param identifier: user input run name
    :param comp: a list of comp(s)
    :param ims: a list of im(s)
    :param period:
    :param geom_only
    :param simple_output
    :return:output result into csvs
    """

    function = 'write_result'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_result_dict.P'), 'wb') as save_file:
            pickle.dump(result_dict, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_output_folder.P'), 'wb') as save_file:
            pickle.dump(output_folder, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_identifier.P'), 'wb') as save_file:
            pickle.dump(identifier, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_comp.P'), 'wb') as save_file:
            pickle.dump(comp, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_ims.P'), 'wb') as save_file:
            pickle.dump(ims, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_period.P'), 'wb') as save_file:
            pickle.dump(period, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_geom_only.P'), 'wb') as save_file:
            pickle.dump(geom_only, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_simple_output.P'), 'wb') as save_file:
            pickle.dump(simple_output, save_file)
        data_taken[function] = True

    output_path = get_result_filepath(output_folder, identifier, ".csv")
    header = get_header(ims, period)
    comp_name, comps = get_comp_name_and_list(comp, geom_only)

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
                    "{}{}.csv".format(station.decode(), comp_name),
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
    :param type: user input
    :param version: user input
    :return:
    """
    date = datetime.now().strftime("%Y%m%d_%H%M%S")

    function = 'generate_metadata'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_output_folder.P'), 'wb') as save_file:
            pickle.dump(output_folder, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_identifier.P'), 'wb') as save_file:
            pickle.dump(identifier, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_rupture.P'), 'wb') as save_file:
            pickle.dump(rupture, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_run_type.P'), 'wb') as save_file:
            pickle.dump(run_type, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_version.P'), 'wb') as save_file:
            pickle.dump(version, save_file)
        data_taken[function] = True

    output_path = get_result_filepath(output_folder, identifier, "_imcalc.info")
    with open(output_path, "w") as meta_file:
        meta_writer = csv.writer(meta_file, delimiter=",", quotechar="|")
        meta_writer.writerow(["identifier", "rupture", "type", "date", "version"])
        meta_writer.writerow([identifier, rupture, run_type, date, version])


def get_comp_help():
    """
    :return: a help message for input component arg
    """
    ret_val = "Available compoents are: {},ellipsis. ellipsis contains all {} components. Default is ellipsis".format(
            ",".join(list(EXT_IDX_DICT.keys())), len(list(EXT_IDX_DICT.keys()))
        )

    function = 'get_comp_help'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_ret_val.P'), 'wb') as save_file:
            pickle.dump(ret_val, save_file)
        data_taken[function] = True
    return (ret_val)


def get_im_or_period_help(default_values, im_or_period):
    """
    :param default_values: predefined constants
    :param im_or_period: should be either string "im" or string "period"
    :return: a help message for input component arg
    """
    function = 'get_im_or_period_help'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_default_values.P'), 'wb') as save_file:
            pickle.dump(default_values, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_im_or_period.P'), 'wb') as save_file:
            pickle.dump(im_or_period, save_file)

    ret_val = "Available and default {}s are: {}".format(
        im_or_period, ",".join(str(v) for v in default_values)
    )

    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_ret_val.P'), 'wb') as save_file:
            pickle.dump(ret_val, save_file)
        data_taken[function] = True
    return ret_val


def validate_input_path(parser, arg_input, arg_file_type):
    """
    validate input path
    :param parser:
    :param arg_input:
    :param arg_file_type:
    :return:
    """

    function = 'validate_input_path'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_arg_input.P'), 'wb') as save_file:
            pickle.dump(arg_input, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_arg_file_type.P'), 'wb') as save_file:
            pickle.dump(arg_file_type, save_file)
        data_taken[function] = True

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


def validate_comp(parser, arg_comp):
    """
    returns validated user input if pass the validation else raise parser error
    :param parser:
    :param arg_comp: user input
    :return: validated comp, only_geom flag
    """

    function = 'validate_comp'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_arg_comp.P'), 'wb') as save_file:
            pickle.dump(arg_comp, save_file)

    comp = arg_comp
    available_comps = list(EXT_IDX_DICT.keys())
    if comp not in available_comps and comp != "ellipsis":
        function = 'get_comp_help'
        ret_val = get_comp_help()
        if not data_taken[function]:
            with open(os.path.join(test_data_save_dir, REALISATION, function + '_return_value.P'), 'wb') as save_file:
                pickle.dump(ret_val, save_file)
        parser.error("please enter a valid comp name. {}".format(ret_val))
    geom_only = (
        False
    )  # when only geom is needed, should be treated as ellipsis but only output geom to csv
    if comp == "geom":
        comp = "ellipsis"
        geom_only = True

    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_comp.P'), 'wb') as save_file:
            pickle.dump(comp, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_geom_only.P'), 'wb') as save_file:
            pickle.dump(geom_only, save_file)
        data_taken[function] = True

    return comp, geom_only


def validate_im(parser, arg_im):
    """
    returns validated user input if pass the validation else raise parser error
    :param parser:
    :param arg_im: input
    :return: validated im(s) in a list
    """
    function = 'validate_im'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_arg_im.P'), 'wb') as save_file:
            pickle.dump(arg_im, save_file)

    im = arg_im
    if im != IMS:
        for m in im:
            if m not in IMS:
                parser.error(
                    "please enter valid im measure name. {}".format(
                        get_im_or_period_help(IMS, "IM")
                    )
                )

    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_im.P'), 'wb') as save_file:
            pickle.dump(im, save_file)
        data_taken[function] = True
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
    function = 'validate_period'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_arg_period.P'), 'wb') as save_file:
            pickle.dump(arg_period, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_arg_extended_period.P'), 'wb') as save_file:
            pickle.dump(arg_extended_period, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_im.P'), 'wb') as save_file:
            pickle.dump(im, save_file)

    period = np.array(arg_period, dtype="float64")

    if arg_extended_period:
        period = np.unique(np.append(period, EXT_PERIOD))

    if (arg_extended_period or period.any()) and "pSA" not in im:
        parser.error(
            "period or extended period must be used with pSA, but pSA is not in the "
            "IM measures entered"
        )

    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_period.P'), 'wb') as save_file:
            pickle.dump(period, save_file)
        data_taken[function] = True
    return period


def get_steps(input_path, nps, total_stations):
    """
    :param input_path: user input file/dir path
    :param nps: number of processes
    :param total_stations: total number of stations
    :return: number of stations per iteration/batch
    """
    function = 'get_steps'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_input_path.P'), 'wb') as save_file:
            pickle.dump(input_path, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_nps.P'), 'wb') as save_file:
            pickle.dump(nps, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_total_stations.P'), 'wb') as save_file:
            pickle.dump(total_stations, save_file)

    estimated_mem = os.stat(input_path).st_size * MEM_FACTOR
    available_mem = nps * MEM_PER_CORE
    batches = np.ceil(np.divide(estimated_mem, available_mem))
    steps = int(np.floor(np.divide(total_stations, batches)))
    if steps == 0:
        steps = total_stations

    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, function + '_steps.P'), 'wb') as save_file:
            pickle.dump(steps, save_file)
        data_taken[function] = True

    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path", help="path to input bb binary file eg./home/melody/BB.bin"
    )
    parser.add_argument(
        "file_type",
        choices=["a", "b"],
        help="Please type 'a'(ascii) or 'b'(binary) to indicate the type of input file",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default=OUTPUT_PATH,
        help="path to output folder that stores the computed measures.Folder name must "
        "not be inclusive.eg.home/tt/. Default to /home/$user/",
    )
    parser.add_argument(
        "-i",
        "--identifier",
        default=RUNNAME_DEFAULT,
        help="Please specify the unique runname of the simulation. "
        "eg.Albury_HYP01-01_S1244",
    )
    parser.add_argument(
        "-r",
        "--rupture",
        default="unknown",
        help="Please specify the rupture name of the simulation. eg.Albury",
    )
    parser.add_argument(
        "-t",
        "--run_type",
        choices=["s", "o", "u"],
        default="u",
        help="Please specify the type of the simrun. Type 's'(simulated) or "
        "'o'(observed) or 'u'(unknown)",
    )
    parser.add_argument(
        "-v",
        "--version",
        default="XXpY",
        help="Please specify the version of the simulation. eg.18p4",
    )
    parser.add_argument(
        "-m",
        "--im",
        nargs="+",
        default=IMS,
        help="Please specify im measure(s) separated by a space(if more than one). "
        "eg: PGV PGA CAV. {}".format(get_im_or_period_help(IMS, "IM")),
    )
    parser.add_argument(
        "-p",
        "--period",
        nargs="+",
        default=BSC_PERIOD,
        type=float,
        help="Please provide pSA period(s) separated by a space. eg: "
        "0.02 0.05 0.1. {}".format(get_im_or_period_help(BSC_PERIOD, "period")),
    )
    parser.add_argument(
        "-e",
        "--extended_period",
        action="store_true",
        help="Please add '-e' to indicate the use of extended(100) pSA periods. "
        "Default not using",
    )
    parser.add_argument(
        "-n",
        "--station_names",
        nargs="+",
        help="Please provide a station name(s) separated by a space. eg: 112A 113A",
    )
    parser.add_argument(
        "-c",
        "--component",
        type=str,
        default="ellipsis",
        help="Please provide the velocity/acc component(s) you want to "
        "calculate eg.geom. {}".format(get_comp_help()),
    )
    parser.add_argument(
        "-np",
        "--process",
        default=2,
        type=int,
        help="Please provide the number of processors. Default is 2",
    )
    parser.add_argument(
        "-s",
        "--simple_output",
        action="store_true",
        help="Please add '-s' to indicate if you want to output the big summary csv "
        "only(no single station csvs). Default outputting both single station "
        "and the big summary csvs",
    )
    parser.add_argument(
        "-u",
        "--units",
        choices=["cm/s^2", "g"],
        default="g",
        help="The units that input acceleration files are in",
    )

    args = parser.parse_args()

    validate_input_path(parser, args.input_path, args.file_type)

    file_type = FILE_TYPE_DICT[args.file_type]

    run_type = META_TYPE_DICT[args.run_type]

    comp, geom_only = validate_comp(parser, args.component)

    im = validate_im(parser, args.im)

    period = validate_period(parser, args.period, args.extended_period, im)

    # Create output dir
    utils.setup_dir(args.output_path)
    if not args.simple_output:
        utils.setup_dir(os.path.join(args.output_path, OUTPUT_SUBFOLDER))

    # multiprocessor
    compute_measures_multiprocess(
        args.input_path,
        file_type,
        geom_only,
        wave_type=None,
        station_names=args.station_names,
        ims=im,
        comp=comp,
        period=period,
        output=args.output_path,
        identifier=args.identifier,
        rupture=args.rupture,
        run_type=run_type,
        version=args.version,
        process=args.process,
        simple_output=args.simple_output,
        units=args.units,
    )

    print("Calculations are outputted to {}".format(args.output_path))
    print(data_taken)


if __name__ == "__main__":
    main()
