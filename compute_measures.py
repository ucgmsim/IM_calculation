#TODO check correctness of nd calc
import sys
import os
import argparse

import numpy as np
import im_calculations
import read_waveform

sys.path.append('../qcore/qcore/')
import timeseries

G = 981.0
OUTPUT = 'computed_measures'
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


def compute_measures(input_path, file_type, wave_type, station_name=None, ims=IMS, comp=None, period=im_calculations.BSC_PERIOD, extended_period=False, meta_data=None, output=OUTPUT):
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
            value = im_calculations.get_max_nd(velocities)
            print("pgv", value)

        if im == "PGA":
            value = im_calculations.get_max_nd(accelerations)
            print("pga",value)

        if im == "pSA":
            value = im_calculations.get_spectral_acceleration_nd(accelerations, period, waveform.NT, DT, extended_period)
            print("psa",value)

        if im == "Ds595":
            value = im_calculations.getDs_nd(DT, accelerations, 5, 95)
            print("ds595",value)

        if im == "Ds575":
            value = im_calculations.getDs_nd(DT, accelerations, 5, 75)
            print("ds575",value)

        if im == "AI":
            value = im_calculations.get_arias_intensity_nd(accelerations, G, times)
            print("ai", value)

        if im == "CAV":
            value = im_calculations.get_cumulative_abs_velocity_nd(accelerations, times)
            print("cav", value)
        #
        if im == "MMI":
            velocities = timeseries.acc2vel(accelerations, DT)
            value = im_calculations.calculate_MMI(velocities)
            print("mmi",value)

        if value.any() or value:
            if im == "pSA":
                result["pSA"] = (period, value)
            else:
                result[im] = value
    print(result)
    return result





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',  help='path to input bb binary file eg./home/melody/BB.bin')
    parser.add_argument('-o', '--output', default='computed_measures',help='path to output folder that stores the computed measures. Default to /computed_measures/')
    parser.add_argument('-m', '--im', nargs='+', default='PGV PGA CAV AI Ds575 Ds595 pSA', help='Please specify im measure(s) seperated by a space(if more than one). eg: PGV PGA CAV. Available and default measures are: PGV, PGA, CAV, AI, Ds575, Ds595, pSA')
    parser.add_argument('-p', '--period', nargs='+', default= '0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.75 1.0 2.0 3.0 4.0 5.0 7.5 10.0', help='Please provide pSA period(s) separated by a space. eg: 0.02 0.05 0.1. Available and default periods are:0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.75 1.0 2.0 3.0 4.0 5.0 7.5 10.0' )
    parser.add_argument('-e','--extended_period', action='store_true', help="Please add '-e' to indicate the use of extended(100) pSA periods. Default not using")
    parser.add_argument('-n','--station_name',help='Please provide a station name. eg: 112A')
    parser.add_argument('-c', '--component', help='Please provide the velocity/acc component(s) you want to calculate eperated by a spave. eg.000 090 ver')
    parser.add_argument('-a', '--ascii', action='store_true', help="Please add '-a' to indicate the type of input file is ascii")
    parser.add_argument('-b', '--binary', action='store_true', help="Please add '-b' to indicate the type of input file is binary")
    args = parser.parse_args()

    if not (args.ascii or args.binary):
        parser.error("Please type either '-a' or '-b' to indicate the type of input file is ascii or binary")

    if args.ascii and args.binary:
        parser.error("'-a' and '-b' option can not be used together. Please type either '-a' or '-b' to indicate the type of input file is ascii or binary")

    if args.ascii:
        file_type = 'standard'
    elif args.binary:
        file_type = 'binary'

    compute_measures(args.input_path, file_type, wave_type=None, station_name='112A', ims=IMS, comp=Ellipsis, period=im_calculations.BSC_PERIOD,
                     extended_period=False, meta_data=None, output=OUTPUT)



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