# TODO make getDs_nd work for 1d ;Modularize; Tidy up

"""command:
   python compute_measures.py /home/vap30/scratch/2/BB.bin b
   python compute_measures.py /home/vap30/scratch/2/BB.bin b -p 0.02 -e -n 112A -c 090
"""

import sys
import os
import errno
import csv
import datetime
import argparse
import numpy as np
import im_calculations
import read_waveform
from qcore import timeseries
from multiprocessing import Pool


G = 981.0
OUTPUT_FOLDER = 'computed_measures_3'
OUTPUT_SUBFOLDER = 'stations_3'
IMS = ['PGV', 'PGA', 'CAV', 'AI', 'Ds575', 'Ds595', 'MMI', 'pSA']
EXT_DICT = {'090': 0, '000': 1, 'ver': 2}
EXT_DICT2 = {0: '090', 1: '000', 2: 'ver'}


def mkdir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def compute_measures_multiprocess(input_path, file_type, wave_type, station_names, ims=IMS, comp=None, period=None, meta_data=None,
                     output=OUTPUT_FOLDER, process=2):
    """
    using multiprocesses to computer measures.
    Calls compute_measure_single() to compute measures for a single station
    wWite results tp csvs
    :param input_path:
    :param file_type:
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
    all_result_dict = {}
    comp_obj = comp
    if comp == 'ellipsis':
        comp_obj = Ellipsis
    waveforms = read_waveform.read_file(input_path, station_names, comp_obj, wave_type=wave_type, file_type=file_type)
    array_params = []
    for waveform in waveforms:
        print("waveforms are",waveforms)
        array_params.append((waveform, ims, comp, period))
    print("arry params",array_params)
    p = Pool(process)

    result_list = p.map(compute_measure_single, array_params)
    print("result list",result_list)

    for result_dict in result_list:
        all_result_dict.update(result_dict)

    write_result(all_result_dict, output, comp, ims, period)


def compute_measure_single((waveform, ims, comp, period)):
    """
    waveform: a single tuple that contains (waveform_acc,waveform_vel)
    Compute measures for a single station
    :return: {result[station_name]: {[im]: value or (period,value}}
    """
    result = {}
    waveform_acc, waveform_vel = waveform
    accelerations = waveform_acc.values
    # print(accelerations.shape)
    DT = waveform_acc.DT
    # print("pppp",period)
    times = waveform_acc.times
    station_name = waveform_acc.station_name
    result[station_name] = {}

    if comp == 'ellipsis':
        comp = Ellipsis

    for im in ims:
        if im == 'PGV':
            print("Afdsafsa", waveform_vel)
            value = im_calculations.get_max_nd(waveform_vel.values)
            print("pgv", value)
            result[station_name][im] = value

        if im == "PGA":
            value = im_calculations.get_max_nd(accelerations)
            print("pga", value)
            result[station_name][im] = value

        if im == "pSA":
            value = im_calculations.get_spectral_acceleration_nd(accelerations, period, waveform_acc.NT, DT)
            print("psa", value)
            result[station_name]["pSA"] = (period, value)

        if im == "Ds595":
            value = im_calculations.getDs_ugly(comp, DT, accelerations, 5, 95)
            print("ds595", value)
            result[station_name][im] = value

        if im == "Ds575":
            value = im_calculations.getDs_ugly(comp, DT, accelerations, 5, 75)
            print("ds575", value)
            result[station_name][im] = value

        if im == "AI":
            value = im_calculations.get_arias_intensity_nd(accelerations, G, times)
            print("ai", value)
            result[station_name][im] = value

        if im == "CAV":
            value = im_calculations.get_cumulative_abs_velocity_nd(accelerations, times)
            print("cav", value)
            result[station_name][im] = value
        #
        if im == "MMI":
            value = im_calculations.calculate_MMI_nd(waveform_vel.values)
            print("mmi", value)
            result[station_name][im] = value
    return result


def compute_measures(input_path, file_type, wave_type, station_names, ims=IMS, comp=None, period=None, meta_data=None,
                     output=OUTPUT_FOLDER):
    # TODO tear down the big func, make it more modular
    """

    :param input_path:
    :param file_type:
    :param wave_type:
    :param station_names:
    :param ims:
    :param comp:
    :param period:
    :param meta_data:
    :param output:
    :return: {result[station_name]: {[im]: value or (period,value}}
    """
    if comp=='ellipsis':
        comp = Ellipsis

    waveforms = read_waveform.read_file(input_path, station_names, comp, wave_type=wave_type, file_type=file_type)
    result = {}

    for waveform_acc, waveform_vel in waveforms:
        accelerations = waveform_acc.values
        # print(accelerations.shape)
        DT = waveform_acc.DT
        # print("pppp",period)
        times = waveform_acc.times
        station_name = waveform_acc.station_name
        result[station_name] = {}

        for im in ims:
            if im == 'PGV':
                print("Afdsafsa", waveform_vel)
                value = im_calculations.get_max_nd(waveform_vel.values)
                print("pgv", value)
                result[station_name][im] = value

            if im == "PGA":
                value = im_calculations.get_max_nd(accelerations)
                print("pga", value)
                result[station_name][im] = value

            if im == "pSA":
                value = im_calculations.get_spectral_acceleration_nd(accelerations, period, waveform_acc.NT, DT)
                print("psa", value)
                result[station_name]["pSA"] = (period, value)

            if im == "Ds595":
                value = im_calculations.getDs_ugly(comp, DT, accelerations, 5, 95)
                print("ds595", value)
                result[station_name][im] = value

            if im == "Ds575":
                value = im_calculations.getDs_ugly(comp, DT, accelerations, 5, 75)
                print("ds575", value)
                result[station_name][im] = value

            if im == "AI":
                value = im_calculations.get_arias_intensity_nd(accelerations, G, times)
                print("ai", value)
                result[station_name][im] = value

            if im == "CAV":
                value = im_calculations.get_cumulative_abs_velocity_nd(accelerations, times)
                print("cav", value)
                result[station_name][im] = value
            #
            if im == "MMI":
                value = im_calculations.calculate_MMI_nd(waveform_vel.values)
                print("mmi", value)
                result[station_name][im] = value

                # if value.any() or value:
                #     if im == "pSA":
                #         result["pSA"] = (period, value)
                #     else:
                #         result[im] = value
    # print(result)
    return result


def compute_measures_on_array((input_path, file_type, wave_type, station_names, ims, comp, period, meta_data,
                     output)):
    return compute_measures(input_path, file_type, wave_type, station_names, ims, comp, period, meta_data, output)



def write_result(result_dict, outputfolder, comp, ims, period):
    # TODO  modularize repeated code
    """

    :param result_dict:
    :param outputfolder:
    :param comp:
    :param ims:
    :param period:
    :return:output result into csvs
    """
    output_path = os.path.join(outputfolder, 'IM_sim_master_{}.csv'.format(str(datetime.datetime.now())))

    row1 = ['station', 'component']
    psa_names = []
    for im in ims:
        if im == 'pSA':
            for p in period:
                psa_names.append('pSA_{}'.format(p))
            row1 += psa_names
        else:
            row1.append(im)

    with open(output_path, 'a') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|')
        csv_writer.writerow(row1)
        stations = result_dict.keys()
        # print("ssss",stations)
        if comp != Ellipsis and comp != 'ellipsis':
            for station in stations:
                station_csv = os.path.join(outputfolder, OUTPUT_SUBFOLDER, '{}_{}.csv'.format(station, EXT_DICT2[comp]))
                print("scsvs s ", station_csv)
                with open(station_csv, 'wb') as sub_csv_file:
                    sub_csv_writer = csv.writer(sub_csv_file, delimiter=',', quotechar='|')
                    sub_csv_writer.writerow(row1)
                    result_row = [station, EXT_DICT2[comp]]
                    # print("start result")
                    for im in ims:
                        if im != 'pSA':
                            result_row.append(result_dict[station][im])
                            # print("result row is ", result_row)
                        else:
                            # print(result_dict[station][im][1][0])
                            result_row += result_dict[station][im][1].tolist()
                            # print("result row is ",result_row)
                    sub_csv_writer.writerow(result_row)
                    csv_writer.writerow(result_row)
        else:
            for station in stations:
                station_csv = os.path.join(outputfolder, OUTPUT_SUBFOLDER, '{}_Ellipsis.csv'.format(station))
                with open(station_csv, 'wb') as sub_csv_file:
                    sub_csv_writer = csv.writer(sub_csv_file, delimiter=',', quotechar='|')
                    sub_csv_writer.writerow(row1)
                    for i in range(3):
                        result_row = [station, EXT_DICT2[i]]
                        for im in ims:
                            if im != 'pSA':
                                result_row.append(result_dict[station][im][i])
                            else:
                                result_row += result_dict[station][im][1][i].tolist()
                        sub_csv_writer.writerow(result_row)
                        csv_writer.writerow(result_row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='path to input bb binary file eg./home/melody/BB.bin')
    parser.add_argument('-o', '--output', default=OUTPUT_FOLDER,
                        help='path to output folder that stores the computed measures. Default to /computed_measures/')
    parser.add_argument('-m', '--im', nargs='+', default=IMS,
                        help='Please specify im measure(s) seperated by a space(if more than one). eg: PGV PGA CAV. Available and default measures are: PGV, PGA, CAV, AI, Ds575, Ds595, pSA')
    parser.add_argument('-p', '--period', nargs='+', default=im_calculations.BSC_PERIOD,
                        help='Please provide pSA period(s) separated by a space. eg: 0.02 0.05 0.1. Available and default periods are:0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.75 1.0 2.0 3.0 4.0 5.0 7.5 10.0')
    parser.add_argument('-e', '--extended_period', action='store_true',
                        help="Please add '-e' to indicate the use of extended(100) pSA periods. Default not using")
    parser.add_argument('-n', '--station_names', nargs='+',
                        help='Please provide a station name(s) seperated by a space. eg: 112A 113A')
    parser.add_argument('-c', '--component', default='ellipsis',
                        help='Please provide the velocity/acc component(s) you want to calculate eperated by a spave. eg.000 090 ver')
    parser.add_argument('file_type', choices=['a', 'b'],
                        help="Please type 'a'(ascii) or 'b'(binary) to indicate the type of input file")
    parser.add_argument('-np', '--process', default=2, type=int, help='Please provide the number of processers')

    # parser.add_argument('-b', '--binary', action='store_true', help="Please add '-b' to indicate the type of input file is binary")
    args = parser.parse_args()

    if args.file_type == 'a':
        file_type = 'standard'
    elif args.file_type == 'b':
        file_type = 'binary'

    period = args.period
    if isinstance(period[0], str):
        period = np.array(args.period, dtype='float64')

    print("period", period)

    comp = args.component
    if comp != 'ellipsis' and comp not in EXT_DICT.values():
        comp = EXT_DICT[args.component.lower()]

    im = args.im
    # print("input im")
    if isinstance(im, str):
        im = args.im.strip().split()
        print("input ims are:", im)

    station_names = args.station_names
    print("args station name", station_names)

    extended_period = args.extended_period
    if extended_period:
        print("flag set")
        period = np.append(period, im_calculations.EXT_PERIOD)
        print("perioddddd", period)

    if (args.extended_period or period.any()) and 'pSA' not in args.im:
        parser.error("period or extended period must be used with pSA, but pSA is not in the IM mesaures entered")

    mkdir(args.output)

    mkdir(os.path.join(args.output, OUTPUT_SUBFOLDER))

    # result_dict = compute_measures(args.input_path, file_type, wave_type=None, station_names=station_names, ims=im,comp=comp, period=period, meta_data=None, output=OUTPUT_FOLDER)
    #
    #
    #
    # print("ccccc",comp)
    # write_result(result_dict, args.output, comp, im, period)

    compute_measures_multiprocess(args.input_path, file_type, wave_type=None, station_names=station_names, ims=im, comp=comp, period=period, meta_data=None, output=OUTPUT_FOLDER, process=args.process)


# if comp and comp != Ellipsis:
#     try:
#         value = im_calculations.getDs(DT, accelerations, 5, 95)
#     except ValueError:
#         sys.exit("Please check if you've entered a correct single ground motion component")
# else:
#     values = []
#     for i in range(3):
#         single_comp = im_calculations.getDs(DT, accelerations[:, i], 5, 95)
#         values.append(single_comp)
#     value = values


# def calc_nd_array(comp, oned_calc_func, extra_args):
#     if comp and comp != Ellipsis:
#         try:
#             value = oned_calc_func(*extra_args)
#         except ValueError:
#             sys.exit("Please check if you've entered a correct single ground motion component")
#     else:
#         values = []
#         for i in range(3):
#             single_comp = oned_calc_func(extra_args)
#             values.append(single_comp)
#         value = values
#     return value
# ('arry params', [((<read_waveform.Waveform instance at 0x7f1a919a6908>, <read_waveform.Waveform instance at 0x7f1a919a6d40>), ['PGV', 'PGA', 'CAV', 'AI', 'Ds575', 'Ds595', 'MMI', 'pSA'], 'ellipsis', array([  0.02,   0.05,   0.1 ,   0.2 ,   0.3 ,   0.4 ,   0.5 ,   0.75,
#          1.  ,   2.  ,   3.  ,   4.  ,   5.  ,   7.5 ,  10.  ])), ((<read_waveform.Waveform instance at 0x7f1a919a6f80>, <read_waveform.Waveform instance at 0x7f1a919a6fc8>), ['PGV', 'PGA', 'CAV', 'AI', 'Ds575', 'Ds595', 'MMI', 'pSA'], 'ellipsis', array([  0.02,   0.05,   0.1 ,   0.2 ,   0.3 ,   0.4 ,   0.5 ,   0.75,
#          1.  ,   2.  ,   3.  ,   4.  ,   5.  ,   7.5 ,  10.  ]))])
