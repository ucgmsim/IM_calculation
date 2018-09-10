"""
python process_csv.py /rcc/home/projects/quakecore/cybershake/v18p6/IMs/ ~/non_uniform_whole_nz_with_real_stations-hh400_v18p6.ll -o ~/v18p6_summary_im/
"""

import os
import glob
import sys
import argparse
from qcore import shared, utils

OUTFILE = "summary_im3.csv"
ERROR_LOG = "err_log.txt"


def log_err(err, err_log_path):
    """
    print and write error to a log file
    :param err: err msg string
    :param err_log_path: path to err log file
    :return:
    """
    print(err)
    with open(err_log_path, 'a') as log:
        log.write("{}\n".format(err))


def get_coords_dict(file_path):
    """
    read a rrup or .ll file, return a coords dict
    :param file_path: path to rrup.csv or staiton.ll
    :return: dict {station_name: (lon, lat)}
    """
    coords_dict = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()
        try:
            for line in lines:
                if '.ll' in file_path:
                    lon, lat, station_name = line.strip().split()
                else:
                    station_name, lon, lat, _, _, _ = line.strip().split(',')
                coords_dict[station_name] = (lon, lat)
        except ValueError:
            sys.exit("Check column numbers in {}".format(file_path))

    return coords_dict


def get_coords(station_name, coords_dict, err_log_path):
    """
    get coords from coords dict
    :param station_name:
    :param coords_dict: {station_name: (lon, lat)}
    :return: coords string
    """
    try:
        lon, lat = coords_dict[station_name]
        return '{},{}'.format(lon, lat)
    except KeyError:
        err = "{} does not exist in the rrup or station file that you provided".format(station_name)
        log_err(err, err_log_path)
        return None


def write_csv(out_dir, faults_dir, staion_filepath):
    """
    write summary csv for a whole simulation
    :param out_dir: output dir for summary csv
    :param faults_dir: dir containing all faults.eg'/rcc/home/projects/quakecore/cybershake/v18p6/IMs/'
    :param staion_filepath: path to a station or rrup file
    :return:
    """
    if out_dir is None:
        out_dir = os.path.join(faults_dir, '..')

    out_path = os.path.join(out_dir, OUTFILE)

    err_log_path = os.path.join(out_dir, ERROR_LOG)

    coords_dict = get_coords_dict(staion_filepath)

    header_written = False

    with open(out_path, 'w') as outfile:
        for fault in os.listdir(faults_dir):
            im_dir = os.path.join(faults_dir, fault, 'IM_calc')
            if os.path.isdir(im_dir):
                for realization in os.listdir(im_dir):
                    realization_dir = os.path.join(im_dir, realization)
                    if os.path.isdir(realization_dir):
                        csv_list = glob.glob1(realization_dir, '*.csv')
                        if len(csv_list) == 1:
                            csv_path = os.path.join(realization_dir, csv_list[0])
                            try:
                                with open(csv_path, 'r') as csv_file:
                                    lines = csv_file.readlines()
                                    header = lines[0].strip()
                                    if not header_written:
                                        headers = header.split(',')
                                        outfile.write("{},lon,lat,{},label\n".format(headers[0], ','.join(headers[1:])))
                                        header_written = True
                                        header_sample = header
                                    else:
                                        assert header == header_sample
                                    for line in lines[1:]:
                                        comps = line.strip().split(',')
                                        station = comps[0]
                                        value_string = ",".join(comps[1:])
                                        coords = get_coords(station, coords_dict, err_log_path)
                                        if not shared.is_virtual_station(station):
                                            outfile.write("{},{},{},{}\n".format(station, coords, value_string, realization))
                            except Exception as e:
                                err = "Error opening {}\n{}".format(csv_path, e)
                                log_err(err, err_log_path)
                                continue
                        else:
                            err = "{} does not have a csv file".format(realization_dir)
                            log_err(err, err_log_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('faults_dir', help="path dir containing all faults, eg'/rcc/home/projects/quakecore/cybershake/v18p6/IMs/'")
    parser.add_argument('rrup_or_station_filepath', help='path to inpurt rrup_csv/station_ll file path')
    parser.add_argument('-o', '--out_dir', help='path to store output csv')
    args = parser.parse_args()

    assert os.path.isdir(args.faults_dir)
    assert os.path.isfile(args.rrup_or_station_filepath)

    if args.out_dir is not None:
        utils.setup_dir(args.out_dir)

    write_csv(args.out_dir, args.faults_dir, args.rrup_or_station_filepath)


if __name__ == '__main__':
    main()
