#TODO check correctness of nd calc
import sys
import os
import errno
import csv
import argparse
import numpy as np
import im_calculations
import read_waveform
from qcore import timeseries

G = 981.0
OUTPUT_FOLDER = 'computed_measures'
IMS = 'PGV PGA CAV AI Ds575 Ds595 pSA MMI'


def calc_nd_array(comp, oned_calc_func, extra_args):
    if comp and comp != Ellipsis:
        try:
            value = oned_calc_func(*extra_args)
        except ValueError:
            sys.exit("Please check if you've entered a correct single ground motion component")
    else:
        values = []
        for i in range(3):
            single_comp = oned_calc_func(extra_args)
            values.append(single_comp)
        value = values
    return value


def get_acc(velocities, DT):
    try:
        accelerations = timeseries.vel2acc(velocities,DT)
    except ValueError:
        print("value error")
        accelerations = timeseries.vel2acc3d(velocities, DT)
    return accelerations


def mkdir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def compute_measures(input_path, file_type, wave_type, station_name=None, ims=IMS, comp=None, period=im_calculations.BSC_PERIOD, extended_period=False, meta_data=None, output=OUTPUT_FOLDER):
    waveform = read_waveform.read_file(input_path, station_name, comp, wave_type=wave_type, file_type=file_type)
    accelerations = waveform.values
    print(accelerations.shape)
    DT = waveform.DT
    print("pppp",period)
    times = waveform.times

    ims = ims.strip().split()
    print(ims)
    value = None
    result = {}

    for im in ims:
        if im == 'PGV':
            velocities = timeseries.acc2vel(accelerations, DT)
            value = im_calculations.get_max(velocities)
            print("pgv", value)
            result[im] = value

        if im == "PGA":
            value = im_calculations.get_max(accelerations)
            print("pga",value)
            result[im] = value

        if im == "pSA":
            value = im_calculations.get_spectral_acceleration_nd(accelerations, period, waveform.NT, DT, extended_period)
            print("psa",value)
            result["pSA"] = (period, value)

        if im == "Ds595":
            value = im_calculations.getDs_nd(DT, accelerations, 5, 95)
            print("ds595",value)
            result[im] = value

        if im == "Ds575":
            value = im_calculations.getDs_nd(DT, accelerations, 5, 75)
            print("ds575",value)
            result[im] = value

        if im == "AI":
            value = im_calculations.get_arias_intensity_nd(accelerations, G, times)
            print("ai", value)
            result[im] = value

        if im == "CAV":
            value = im_calculations.get_cumulative_abs_velocity_nd(accelerations, times)
            print("cav", value)
            result[im] = value
        #
        # if im == "MMI":
        #     velocities = timeseries.acc2vel(accelerations, DT)
        #     value = im_calculations.calculate_MMI(velocities)
        #     print("mmi",value)
        #     result[im] = value

        # if value.any() or value:
        #     if im == "pSA":
        #         result["pSA"] = (period, value)
        #     else:
        #         result[im] = value
    print(result)
    return result


# def write_result(result_dict, output_path, comp, ims, period):
#     psa_names = []
#     for im in ims:
#         if im == 'pSA':
#
#     with open(output_path,'wb') as csv_file:
#         csv_writer = csv.writer(csv_file,delimiter=',', quotechar='|')
#         row1 = ['station','component']
#         if result_dict['pSA']:
#              += 'pSA_{}'.format(p for p in result_dict['pSA'][0])]
#
#
#         csv_writer.writerow([])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',  help='path to input bb binary file eg./home/melody/BB.bin')
    parser.add_argument('-o', '--output', default='computed_measures',help='path to output folder that stores the computed measures. Default to /computed_measures/')
    parser.add_argument('-m', '--im', nargs='+', default='PGV PGA CAV AI Ds575 Ds595 pSA', help='Please specify im measure(s) seperated by a space(if more than one). eg: PGV PGA CAV. Available and default measures are: PGV, PGA, CAV, AI, Ds575, Ds595, pSA')
    parser.add_argument('-p', '--period', nargs='+', default= '0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.75 1.0 2.0 3.0 4.0 5.0 7.5 10.0', help='Please provide pSA period(s) separated by a space. eg: 0.02 0.05 0.1. Available and default periods are:0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.75 1.0 2.0 3.0 4.0 5.0 7.5 10.0' )
    parser.add_argument('-e','--extended_period', action='store_true', help="Please add '-e' to indicate the use of extended(100) pSA periods. Default not using")
    parser.add_argument('-n','--station_name',help='Please provide a station name. eg: 112A')
    parser.add_argument('-c', '--component', help='Please provide the velocity/acc component(s) you want to calculate eperated by a spave. eg.000 090 ver')
    parser.add_argument('file_type', choices=['a', 'b'], help="Please type 'a'(ascii) or 'b'(binary) to indicate the type of input file")
    # parser.add_argument('-b', '--binary', action='store_true', help="Please add '-b' to indicate the type of input file is binary")
    args = parser.parse_args()

    if args.file_type == 'a':
        file_type = 'standard'
    elif args.file_type == 'b':
        file_type = 'binary'

    if args.period:
        period = np.array(args.period.strip().split())

    mkdir(OUTPUT_FOLDER)

    compute_measures(args.input_path, file_type, wave_type=None, station_name='112A', ims=IMS, comp=Ellipsis, period=im_calculations.BSC_PERIOD,
                     extended_period=False, meta_data=None, output=OUTPUT_FOLDER)




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

    def getDerivative(x, fx, order=2):
        """Uses central differences to get the derivative of f(x).  x must be in
        ascending or descending order
        Based on ComputeIMsFromGroundMotionMatFile.m
        Inputs:
            fx - function to be differentiated
            x - co-ordinate to be differentiated with respect to
            order - difference between x values (default=2)
        Outputs:
            dfdx - the derivative of fx wrt x
        """
        n = np.size(x)

        dfdx = np.zeros(n)

        # for the first step use forward differences
        dfdx[0] = (fx[1] - fx[0]) / (x[1] - x[0])

        # for all other steps
        for i in xrange(1, n - 1):
            if order == 1:
                dfdx[i] = (fx[i] - fx[i - 1]) / (x[i] - x[i - 1])
            elif order == 2:
                dfdx[i] = (fx[i + 1] - fx[i - 1]) / (x[i + 1] - x[i - 1])

        # for the last step use back differences
        dfdx[-1] = (fx[-1] - fx[-2]) / (x[-1] - x[-2])

        return dfdx