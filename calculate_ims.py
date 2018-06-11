# TODO when only geom is needed, only 090 and 000 should be calculated
"""
Calculate im values.
Output computed measures to /home/$user/computed_measures if no output path is specified
command:
   python calculate_ims.py test/test_calculate_ims/sample1/input/darfield_ascii/ a
   python calculate_ims.py ../BB.bin b
   python calculate_ims.py ../BB.bin b -o /home/yzh231/ -i Albury_666_999 -r Albury -t s -v 18p3 -n 112A CMZ -m PGV pSA -p 0.02 0.03 -e -c geom -np 2
"""

import os
import csv
import argparse
import getpass
import numpy as np
from collections import OrderedDict
from datetime import datetime
from IM import intensity_measures
from IM import read_waveform
from rrup import pool_wrapper
from qcore import utils
from qcore import timeseries

G = 981.0
IMS = ['PGV', 'PGA', 'CAV', 'AI', 'Ds575', 'Ds595', 'MMI', 'pSA']

EXT_PERIOD = np.logspace(start=np.log10(0.01), stop=np.log10(10.), num=100, base=10)
BSC_PERIOD = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]

IDX_EXT_DICT = OrderedDict([(0, '090'), (1, '000'), (2, 'ver'), (3, 'geom')])
EXT_IDX_DICT = OrderedDict((v, k) for k, v in IDX_EXT_DICT.items())

FILE_TYPE_DICT = {'a': 'ascii', 'b': 'binary'}
META_TYPE_DICT = {'s': 'simulated', 'o': 'observed', 'u': 'unknown'}

OUTPUT_PATH = os.path.join('/home', getpass.getuser())
OUTPUT_SUBFOLDER = 'stations'

RUNNAME_DEFAULT = 'all_station_ims'


def convert_str_comp(comp):
    """
    convert string comp eg '090'/'ellipsis' to int 0/Ellipsis obj
    :param comp: user input
    :return: converted comp
    """
    if comp == 'ellipsis':
        converted_comp = Ellipsis
    else:
        converted_comp = EXT_IDX_DICT[comp]
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
    value_dict = {}
    if converted_comp == Ellipsis:
        comps = EXT_IDX_DICT.keys()
        for c in comps[:-1]:  # excludes geom
            column = EXT_IDX_DICT[c]
            if im == 'pSA':  # pSA returns 2d array
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
        if im == 'MMI':
            value = value.item(0)  # mmi somehow returns a single array instead of a num
        value_dict[comp] = value
    return value_dict


def compute_measure_single((waveform, ims, comp, period)):
    """
    Compute measures for a single station
    :param: a tuple consisting 4 params: waveform, ims, comp, period
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
    converted_comp = convert_str_comp(comp)

    for im in ims:
        if im == 'PGV':
            value = intensity_measures.get_max_nd(velocities)

        if im == "PGA":
            value = intensity_measures.get_max_nd(accelerations)

        if im == "pSA":
            value = intensity_measures.get_spectral_acceleration_nd(accelerations, period, waveform_acc.NT, DT)

        #TODO: Speed up Ds calculations
        if im == "Ds595":
            value = intensity_measures.getDs_nd(DT, accelerations, 5, 95)

        if im == "Ds575":
            value = intensity_measures.getDs_nd(DT, accelerations, 5, 75)

        if im == "AI":
            value = intensity_measures.get_arias_intensity_nd(accelerations, G, times)

        if im == "CAV":
            value = intensity_measures.get_cumulative_abs_velocity_nd(accelerations, times)

        if im == "MMI":
            value = intensity_measures.calculate_MMI_nd(velocities)

        # store a im type values into a dict {comp: np_array/single float}
        # Geometric is also calculated here
        value_dict = array_to_dict(value, comp, converted_comp, im)

        # store value dict into the biggest result dict
        if im == 'pSA':
            result[station_name][im] = (period, value_dict)
        else:
            result[station_name][im] = value_dict

    return result


def compute_measures_multiprocess(input_path, file_type, geom_only, wave_type, station_names, ims=IMS, comp=None,
                                  period=None, output=None, identifier=None, rupture=None, run_type=None, version=None,
                                  process=1):
    """
    using multiprocesses to computer measures.
    Calls compute_measure_single() to compute measures for a single station
    write results to csvs and a .info meta data file
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
    :param type:
    :param version:
    :param process:
    :return:
    """
    converted_comp = convert_str_comp(comp)

    waveforms = read_waveform.read_waveforms(input_path, station_names, converted_comp, wave_type=wave_type,
                                             file_type=file_type)
    array_params = []
    all_result_dict = {}

    for waveform in waveforms:
        array_params.append((waveform, ims, comp, period))

    p = pool_wrapper.PoolWrapper(process)

    result_list = p.map(compute_measure_single, array_params)

    for result in result_list:
        all_result_dict.update(result)

    write_result(all_result_dict, output, identifier, comp, ims, period, geom_only)

    generate_metadata(output, identifier, rupture, run_type, version)


def get_result_filepath(output_folder, arg_identifier, suffix):
    return os.path.join(output_folder, '{}.{}'.format(arg_identifier, suffix))


def get_header(ims, period):
    """
    write header colums for output im_calculations csv file
    :param ims: a list of im measures
    :param period: a list of pSA periods
    :return: header
    """
    header = ['station', 'component']
    psa_names = []

    for im in ims:
        if im == 'pSA':  # only write period if im is pSA.
            for p in period:
                psa_names.append('pSA_{}'.format(p))
            header += psa_names
        else:
            header.append(im)
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
    if geom_only:
        comp_name = '_geom'
        comps = ['geom']

    elif comp == 'ellipsis':
        comp_name = ''
        comps = EXT_IDX_DICT.keys()

    else:
        comp_name = '_{}'.format(comp)
        comps = [comp]

    return comp_name, comps


def write_result(result_dict, output_folder, identifier, comp, ims, period, geom_only):
    """
    write a big csv that contains all calculated im value and single station csvs
    :param result_dict:
    :param output_folder:
    :param identifier: user input run name
    :param comp: a list of comp(s)
    :param ims: a list of im(s)
    :param period:
    :param geom_only
    :return:output result into csvs
    """
    output_path = get_result_filepath(output_folder, identifier, 'csv')

    header = get_header(ims, period)

    comp_name, comps = get_comp_name_and_list(comp, geom_only)

    # big csv containing all stations
    with open(output_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|')
        csv_writer.writerow(header)
        stations = result_dict.keys()

        # sub station csv
        for station in stations:
            station_csv = os.path.join(output_folder, OUTPUT_SUBFOLDER, '{}{}.csv'.format(station, comp_name))
            with open(station_csv, 'wb') as sub_csv_file:
                sub_csv_writer = csv.writer(sub_csv_file, delimiter=',', quotechar='|')
                sub_csv_writer.writerow(header)
                for c in comps:
                    row = [station, c]
                    for im in ims:
                        if im != 'pSA':
                            row.append(result_dict[station][im][c])
                        else:
                            row += result_dict[station][im][1][c].tolist()
                    sub_csv_writer.writerow(row)
                    csv_writer.writerow(row)


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
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = get_result_filepath(output_folder, identifier, 'info')

    with open(output_path, 'w') as meta_file:
        meta_writer = csv.writer(meta_file, delimiter=',', quotechar='|')
        meta_writer.writerow(['identifier', 'rupture', 'type', 'date', 'version'])
        meta_writer.writerow([identifier, rupture, run_type, date, version])


def get_comp_help():
    """
    :return: a help message for input component arg
    """
    return 'Available compoents are: {},ellipsis. ellipsis contains all {} components. Default is ellipsis'.format(','.join(c for c in EXT_IDX_DICT.keys()), len(EXT_IDX_DICT.keys()))


def get_im_or_period_help(default_values, im_or_period):
    """
    :param default_values: predefined constants
    :param im_or_period: should be either string "im" or string "period"
    :return: a help message for input component arg
    """
    return 'Available and default {}s are: {}'.format(im_or_period, ','.join(str(v) for v in default_values))


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

    if arg_file_type == 'b':
        if os.path.isdir(arg_input):
            parser.error('The path should point to a binary file but not a directory. Correct Sample: /home/tt/BB.bin')
    elif arg_file_type == 'a':
        if os.path.isfile(arg_input):
            parser.error('The path should be a directory but not a file. Correct Sample: /home/tt/sims/')


def validate_comp(parser, arg_comp):
    """
    returns validated user input if pass the validation else raise parser error
    :param parser:
    :param arg_comp: user input
    :return: validated comp, only_geom flag
    """
    comp = arg_comp
    available_comps = EXT_IDX_DICT.keys()
    if comp not in available_comps and comp != 'ellipsis':
        parser.error("please enter a valid comp name. {}".format(get_comp_help()))
    geom_only = False  # when only geom is needed, should be treated as ellipsis but only output geom to csv
    if comp == 'geom':
        comp = 'ellipsis'
        geom_only = True
    return comp, geom_only


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
                parser.error('please enter valid im measure name. {}'.format(get_im_or_period_help(IMS, "IM")))
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
    period = arg_period
    extended_period = arg_extended_period

    period = np.array(period, dtype='float64')

    if extended_period:
        period = np.append(period, EXT_PERIOD)

    if (extended_period or period.any()) and 'pSA' not in im:
        parser.error("period or extended period must be used with pSA, but pSA is not in the IM measures entered")

    return period


def mkdir_output(arg_output, arg_identifier):
    """
    create big output dir and sub dir 'stations' inside the big output dir
    :param arg_output:
    :param arg_identifier:
    :return: path to the big output_dir
    """
    output_dir = os.path.join(arg_output, arg_identifier)
    utils.setup_dir(output_dir)
    utils.setup_dir(os.path.join(output_dir, OUTPUT_SUBFOLDER))

    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='path to input bb binary file eg./home/melody/BB.bin')
    parser.add_argument('file_type', choices=['a', 'b'],
                        help="Please type 'a'(ascii) or 'b'(binary) to indicate the type of input file")
    parser.add_argument('-o', '--output_path', default=OUTPUT_PATH,
                        help='path to output folder that stores the computed measures.Folder name must not be inclusive.eg.home/tt/. Default to /home/$user/')
    parser.add_argument('-i', '--identifier', default=RUNNAME_DEFAULT,
                        help='Please specify the unique runname of the simulation. eg.Albury_HYP01-01_S1244')
    parser.add_argument('-r', '--rupture', default='unknown',
                        help='Please specify the rupture name of the simulation. eg.Albury')
    parser.add_argument('-t', '--run_type', choices=['s', 'o', 'u'], default='u',
                        help="Please specify the type of the simrun. Type 's'(simulated) or 'o'(observed) or 'u'(unknown)")
    parser.add_argument('-v', '--version', default='XXpY', help='Please specify the version of the simulation. eg.18p4')
    parser.add_argument('-m', '--im', nargs='+', default=IMS,
                        help='Please specify im measure(s) separated by a space(if more than one). eg: PGV PGA CAV. {}'.format(get_im_or_period_help(IMS, "IM")))
    parser.add_argument('-p', '--period', nargs='+', default=BSC_PERIOD, type=float,
                        help='Please provide pSA period(s) separated by a space. eg: 0.02 0.05 0.1. {}'.format(get_im_or_period_help(BSC_PERIOD, "period")))
    parser.add_argument('-e', '--extended_period', action='store_true',
                        help="Please add '-e' to indicate the use of extended(100) pSA periods. Default not using")
    parser.add_argument('-n', '--station_names', nargs='+',
                        help='Please provide a station name(s) separated by a space. eg: 112A 113A')
    parser.add_argument('-c', '--component', type=str, default='ellipsis',
                        help='Please provide the velocity/acc component(s) you want to calculate eg.geom. {}'.format(get_comp_help()))
    parser.add_argument('-np', '--process', default=2, type=int, help='Please provide the number of processors. Default is 2')

    args = parser.parse_args()

    validate_input_path(parser, args.input_path, args.file_type)

    file_type = FILE_TYPE_DICT[args.file_type]

    run_type = META_TYPE_DICT[args.run_type]

    comp, geom_only = validate_comp(parser, args.component)

    im = validate_im(parser, args.im)

    period = validate_period(parser, args.period, args.extended_period, im)

    output_dir = mkdir_output(args.output_path, args.identifier)

    # multiprocessor
    compute_measures_multiprocess(args.input_path, file_type, geom_only, wave_type=None,
                                  station_names=args.station_names, ims=im, comp=comp, period=period, output=output_dir,
                                  identifier=args.identifier, rupture=args.rupture, run_type=run_type, version=args.version,
                                  process=args.process)

    print("Calculations are outputted to {}".format(output_dir))


if __name__ == '__main__':
    main()