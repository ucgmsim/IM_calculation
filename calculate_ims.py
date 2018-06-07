# TODO when only geom is needed, only 090 and 000 should be calculated
"""
Calculate im values.
Output computed measures to /home/$user/computed_measures if no output path is specified
command:
   python compute_measures.py /home/vap30/scratch/2/BB.bin b
   python compute_measures.py /home/vap30/scratch/2/BB.bin b -p 0.02 -e -n 112A -c 090 -m PGV pSA -np 2
"""

import os
import errno
import csv
import argparse
import getpass
import numpy as np
from collections import OrderedDict
import intensity_measures
import read_waveform
from rrup import pool_wrapper

G = 981.0
IMS = ['PGV', 'PGA', 'CAV', 'AI', 'Ds575', 'Ds595', 'MMI', 'pSA']

EXT_PERIOD = np.logspace(start=np.log10(0.01), stop=np.log10(10.), num=100, base=10)
BSC_PERIOD = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0])

IDX_EXT_DICT = OrderedDict([(0, '090'), (1, '000'), (2, 'ver'), (3, 'geom')])
EXT_IDX_DICT = OrderedDict((v, k) for k, v in IDX_EXT_DICT.items())

FILE_TYPE_DICT = {'a': 'ascii', 'b': 'binary'}

OUTPUT_FOLDER = os.path.join('/home', getpass.getuser(), 'computed_measures')
OUTPUT_SUBFOLDER = 'stations'
ALL_STATION_CSV_FILE = 'all_station_ims.csv'


def mkdir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


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


def array_to_dict(value, comp, im):
    """
    convert a numpy arrary that contains calculated im values to a dict {comp: value}
    :param value:
    :param comp:
    :param im:
    :return: a dict {comp: value}
    """
    converted_comp = convert_str_comp(comp)
    value_dict = OrderedDict()
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
    accelerations = waveform_acc.values
    DT = waveform_acc.DT
    times = waveform_acc.times
    station_name = waveform_acc.station_name
    result[station_name] = {}

    for im in ims:
        # value = [None, None, None]
        if im == 'PGV' and waveform_vel is not None:
            value = intensity_measures.get_max_nd(waveform_vel.values)

        if im == "PGA":
            value = intensity_measures.get_max_nd(accelerations)

        if im == "pSA":
            value = intensity_measures.get_spectral_acceleration_nd(accelerations, period, waveform_acc.NT, DT)

        if im == "Ds595":
            value = intensity_measures.getDs_nd(DT, accelerations, 5, 95)

        if im == "Ds575":
            value = intensity_measures.getDs_nd(DT, accelerations, 5, 75)

        if im == "AI":
            value = intensity_measures.get_arias_intensity_nd(accelerations, G, times)

        if im == "CAV":
            value = intensity_measures.get_cumulative_abs_velocity_nd(accelerations, times)

        if im == "MMI" and waveform_vel is not None:
            value = intensity_measures.calculate_MMI_nd(waveform_vel.values)

        # store a im type values into a dict {comp: np_array/single float}
        value_dict = array_to_dict(value, comp, im)

        # store value dict into the biggest result dict
        if im == 'pSA':
            result[station_name][im] = (period, value_dict)
        else:
            result[station_name][im] = value_dict

    return result



def compute_measures_multiprocess(input_path, file_type, geom_only, wave_type, station_names, ims=IMS, comp=None,
                                  period=None, meta_data=None, output=OUTPUT_FOLDER, process=1):
    """
    using multiprocesses to computer measures.
    Calls compute_measure_single() to compute measures for a single station
    wWite results tp csvs
    :param input_path:
    :param file_type:
    :param geom_only:
    :param wave_type:
    :param station_names:
    :param ims:
    :param comp:
    :param period:
    :param meta_data:
    :param output:
    :param process:
    :return: writes
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

    write_result(all_result_dict, output, comp, ims, period, geom_only)


def get_header(ims, period):
    """
    write header colums for output im_calculations csv file
    :param ims: a list of im measures
    :param period: a list of pSA periods
    :return:
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
    :param comp:
    :param geom_only:
    :param output_folder:
    :return:
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


def write_result(result_dict, output_folder, comp, ims, period, geom_only):
    """
    write a big csv that contains all calculated im value and single station csvs
    :param result_dict:
    :param output_folder:
    :param comp:
    :param ims:
    :param period:
    :param geom_only
    :return:output result into csvs
    """
    output_path = os.path.join(output_folder, ALL_STATION_CSV_FILE)

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


def validate_comp(parser, arg_comp):
    """
    returns validated user input if pass the validation else raise parser error
    :param parser:
    :param arg_comp: user input
    :return: validated comp, only_geom flag
    """
    comp = arg_comp
    if comp not in EXT_IDX_DICT.keys() and comp != 'ellipsis':
        parser.error("please enter a valid comp name. Available compoents are: 090,000,ver,geom,ellipsis. ellipsis contains all 4 components")
    geom_only = False  # when only geom is needed, should be treated as ellipsis but only output geom to csv
    if comp == 'geom':
        comp = 'ellipsis'
        geom_only = True
    return comp, geom_only


def validate_im(parser, arg_im):
    """
    returns validated user input if pass the validation else raise parser error
    :param parser:
    :param arg_im:
    :return: validated im(s) in a list
    """
    im = arg_im
    if isinstance(im, str):
        im = im.strip().split()
        for m in im:
            if m not in IMS:
                parser.error('please enter valid im meausre name. Available and default measures are: PGV, PGA, CAV, AI, Ds575, Ds595, pSA')
    return im


def validate_period(parser, arg_period, arg_extended_period, im):
    """
    returns validated user input if pass the validation else raise parser error
    :param parser:
    :param arg_period:
    :param arg_extended_period:
    :param im: validated im(s) in a list
    :return: period(s) in a numpy arrau
    """
    period = arg_period
    extended_period = arg_extended_period

    if isinstance(period[0], str):
        period = np.array(period, dtype='float64')

    if extended_period:
        period = np.append(period, EXT_PERIOD)

    if (extended_period or period.any()) and 'pSA' not in im:
        parser.error("period or extended period must be used with pSA, but pSA is not in the IM mesaures entered")

    return period


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='path to input bb binary file eg./home/melody/BB.bin')
    parser.add_argument('-o', '--output', default=OUTPUT_FOLDER,
                        help='path to output folder that stores the computed measures. Default to /home/$user/computed_measures')
    parser.add_argument('-m', '--im', nargs='+', default=IMS,
                        help='Please specify im measure(s) seperated by a space(if more than one). eg: PGV PGA CAV. Available and default measures are: PGV, PGA, CAV, AI, Ds575, Ds595, pSA')
    parser.add_argument('-p', '--period', nargs='+', default=BSC_PERIOD,
                        help='Please provide pSA period(s) separated by a space. eg: 0.02 0.05 0.1. Available and default periods are:0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.75 1.0 2.0 3.0 4.0 5.0 7.5 10.0')
    parser.add_argument('-e', '--extended_period', action='store_true',
                        help="Please add '-e' to indicate the use of extended(100) pSA periods. Default not using")
    parser.add_argument('-n', '--station_names', nargs='+',
                        help='Please provide a station name(s) seperated by a space. eg: 112A 113A')
    parser.add_argument('-c', '--component', type=str, default='ellipsis',
                        help='Please provide the velocity/acc component(s) you want to calculate eg.geom. Available compoents are: 090,000,ver,geom,ellipsis. ellipsis contains all 4 components')
    parser.add_argument('file_type', choices=['a', 'b'],
                        help="Please type 'a'(ascii) or 'b'(binary) to indicate the type of input file")
    parser.add_argument('-np', '--process', default=2, type=int, help='Please provide the number of processers')

    args = parser.parse_args()

    file_type = FILE_TYPE_DICT[args.file_type]

    station_names = args.station_names

    comp, geom_only = validate_comp(parser, args.component)

    im = validate_im(parser, args.im)

    period = validate_period(parser, args.period, args.extended_period, im)

    mkdir(args.output)

    mkdir(os.path.join(args.output, OUTPUT_SUBFOLDER))

    # multiprocessor
    compute_measures_multiprocess(args.input_path, file_type, geom_only, wave_type=None, station_names=station_names,
                                  ims=im, comp=comp, period=period, meta_data=None, output=args.output,
                                  process=args.process)

    print("Calculations are outputted to {}".format(OUTPUT_FOLDER))


if __name__ == '__main__':
    main()
