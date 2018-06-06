# TODO MMI calc returns an array of 1 item when only 1 comp is given

"""command:
   python compute_measures.py /home/vap30/scratch/2/BB.bin b
   python compute_measures.py /home/vap30/scratch/2/BB.bin b -p 0.02 -e -n 112A -c 090
"""

import os
import errno
import csv
import argparse
import numpy as np
from collections import OrderedDict
from multiprocessing import Pool
import intensity_measures
import read_waveform


G = 981.0
IMS = ['PGV', 'PGA', 'CAV', 'AI', 'Ds575', 'Ds595', 'pSA']

EXT_PERIOD = np.logspace(start=np.log10(0.01), stop=np.log10(10.), num=100, base=10)
BSC_PERIOD = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0])

IDX_EXT_DICT = OrderedDict([(0, '090'), (1, '000'), (2, 'ver'), (3, 'geom')])
EXT_IDX_DICT = OrderedDict((v, k) for k, v in IDX_EXT_DICT.items())

OUTPUT_FOLDER = 'computed_measures3'  # todo should be outside ims
OUTPUT_SUBFOLDER = 'stations_3'


def mkdir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def convert_str_comp(comp):
    if comp == 'ellipsis':
        converted_comp = Ellipsis
    else:
        converted_comp = EXT_IDX_DICT[comp]
    return converted_comp


def compute_measures_multiprocess(input_path, file_type, wave_type, station_names, ims=IMS, comp=None, period=None,
                                  meta_data=None,
                                  output=OUTPUT_FOLDER, process=1):
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

    converted_comp = convert_str_comp(comp)

    waveforms = read_waveform.read_waveforms(input_path, station_names, converted_comp, wave_type=wave_type, file_type=file_type)
    array_params = []
    all_result_dict = {}

    for waveform in waveforms:
        array_params.append((waveform, ims, comp, period))
    # TODO: use pool wrapper here to avoid this
    if process == 1:
        result_list = []
        for params in array_params:
            result_list.append(compute_measure_single(params))
    else:
        p = Pool(process)

        result_list = p.map(compute_measure_single, array_params)

    for result in result_list:
        all_result_dict.update(result)

    write_result2(all_result_dict, output, comp, ims, period)


def compute_measure_single((waveform, ims, comp, period)):
    """
    waveform: a single tuple that contains (waveform_acc,waveform_vel)
    Compute measures for a single station
    :return: {result[station_name]: {[im]: value or (period,value}}
    """
    converted_comp = convert_str_comp(comp)

    print("afsfad", converted_comp)
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

        value_dict = OrderedDict()  # {comp: np_array/single float}
        if converted_comp == Ellipsis:
            print(im,value)
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
            print(geom_value)
            value_dict[comps[-1]] = geom_value   # now set geom
        else:
            print(im, value)
            print("afsfad",comp)
            # only one comp
            # geom_value = None
            value_dict[comp] = value

        if im == 'pSA':
            result[station_name][im] = (period, value_dict)
        else:
            result[station_name][im] = value_dict
            #TODO DELETE below grey after testing
        # if comp is Ellipsis:
        #     print im, value
        #     print type(value)
        #     if im =='pSA' or (None not in value):
        #         d1 = value[0]
        #         d2 = value[1]
        #         geom_value = intensity_measures.get_geom(d1, d2)
        #     else:
        #         geom_value = None
        #     if im != 'pSA':
        #         value = np.append(value, geom_value)
        #     else:
        #         value.append(geom_value)
        #
        # if im == 'pSA':
        #     result[station_name][im] = (period, value)
        #     # print("result is",result)
        # else:
        #     result[station_name][im] = value
    print(result)
    return result


def compute_measures(input_path, file_type, wave_type, station_names, ims=IMS, comp=None, period=None, meta_data=None,
                     output=OUTPUT_FOLDER):
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
    converted_comp = convert_str_comp(comp)

    result = {}
    waveforms = read_waveform.read_waveforms(input_path, station_names, converted_comp, wave_type=wave_type, file_type=file_type)

    for waveform in waveforms:
        result.update(compute_measure_single((waveform, ims, converted_comp, period)))
    return result


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


def write_result2(result_dict, outputfolder, comp, ims, period):
    # TODO  wrapper sub csv
    """
    :param result_dict:
    :param outputfolder:
    :param comp:
    :param ims:
    :param period:
    :return:output result into csvs
    """
    output_path = os.path.join(outputfolder, 'all_station_ims.csv')

    header = get_header(ims, period)

    with open(output_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|')
        csv_writer.writerow(header)
        stations = result_dict.keys()
        for station in stations:
            station_csv = os.path.join(outputfolder, OUTPUT_SUBFOLDER, '{}_{}.csv'.format(station, comp))
            with open(station_csv, 'wb') as sub_csv_file:
                sub_csv_writer = csv.writer(sub_csv_file, delimiter=',', quotechar='|')
                sub_csv_writer.writerow(header)
                if comp == 'ellipsis':
                    comps = EXT_IDX_DICT.keys()
                else:
                    comps = [comp]
                for c in comps:
                    row = [station, c]
                    for im in ims:
                        if im != 'pSA':
                            row.append(result_dict[station][im][c])
                        else:
                            print("is psa",result_dict[station][im])
                            row += result_dict[station][im][1][c].tolist()
                    sub_csv_writer.writerow(row)
                    csv_writer.writerow(row)



def write_result(result_dict, outputfolder, comp, ims, period):
    # TODO delete write_result after write_result2 is tested
    """

    :param result_dict:
    :param outputfolder:
    :param comp:
    :param ims:
    :param period:
    :return:output result into csvs
    """
    output_path = os.path.join(outputfolder, 'all_station_ims.csv')

    header = get_header(ims, period)

    with open(output_path, 'a') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|')
        csv_writer.writerow(header)
        stations = result_dict.keys()
        if comp != Ellipsis and comp != 'ellipsis':
            for station in stations:
                station_csv = os.path.join(outputfolder, OUTPUT_SUBFOLDER, '{}_{}.csv'.format(station, EXT_IDX_DICT[comp]))
                print("scsvs s ", station_csv)
                with open(station_csv, 'wb') as sub_csv_file:
                    sub_csv_writer = csv.writer(sub_csv_file, delimiter=',', quotechar='|')
                    sub_csv_writer.writerow(header)
                    result_row = [station, EXT_IDX_DICT[comp]]
                    for im in ims:
                        if im != 'pSA':
                            result_row.append(result_dict[station][im])
                        else:
                            result_row += result_dict[station][im][1].tolist()
                    sub_csv_writer.writerow(result_row)
                    csv_writer.writerow(result_row)
        else:
            for station in stations:
                station_csv = os.path.join(outputfolder, OUTPUT_SUBFOLDER, '{}.csv'.format(station))
                with open(station_csv, 'wb') as sub_csv_file:
                    sub_csv_writer = csv.writer(sub_csv_file, delimiter=',', quotechar='|')
                    sub_csv_writer.writerow(header)
                    for i in range(4):
                        result_row = [station, IDX_EXT_DICT[i]]
                        for im in ims:
                            if im != 'pSA':
                                result_row.append(result_dict[station][im][i])
                            else:
                                print("result row psa", result_dict[station][im][1])
                                result_row += result_dict[station][im][1][i].tolist()
                        sub_csv_writer.writerow(result_row)
                        csv_writer.writerow(result_row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='path to input bb binary file eg./home/melody/BB.bin')
    parser.add_argument('-o', '--output', default=OUTPUT_FOLDER,
                        help='path to output folder that stores the computed measures. Default to /computed_measures/')
    parser.add_argument('-m', '--im', nargs='+', default=IMS,
                        help='Please specify im measure(s) seperated by a space(if more than one). eg: PGV PGA CAV. Available and default measures are: PGV, PGA, CAV, AI, Ds575, Ds595, pSA')
    parser.add_argument('-p', '--period', nargs='+', default=BSC_PERIOD,
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

    args = parser.parse_args()

    if args.file_type == 'a':
        file_type = 'standard'
    elif args.file_type == 'b':
        file_type = 'binary'

    period = args.period
    if isinstance(period[0], str):
        period = np.array(args.period, dtype='float64')

    comp = args.component
    # if comp != 'ellipsis' and comp not in EXT_IDX_DICT.values():
    #     comp = EXT_IDX_DICT[args.component.lower()]

    im = args.im
    if isinstance(im, str):
        im = args.im.strip().split()

    station_names = args.station_names

    extended_period = args.extended_period
    if extended_period:
        period = np.append(period, EXT_PERIOD)

    if (args.extended_period or period.any()) and 'pSA' not in args.im:
        parser.error("period or extended period must be used with pSA, but pSA is not in the IM mesaures entered")

    mkdir(args.output)

    mkdir(os.path.join(args.output, OUTPUT_SUBFOLDER))

    # multiprocessor
    compute_measures_multiprocess(args.input_path, file_type, wave_type=None, station_names=station_names, ims=im,
                                  comp=comp, period=period, meta_data=None, output=args.output, process=args.process)


if __name__ == '__main__':
    main()
