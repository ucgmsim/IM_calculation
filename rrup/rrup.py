# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:29:34 2016

@author: rmc84

A collection of functions relating to reading the station and rupture files, taken from the matlab code.
In each function, the name of the original .m function is indicated.
Function and variable names from matlab are mostly preserved.

"""

from math import radians, sin, cos, asin, sqrt, ceil
import sys
import os
import pickle

from qcore.srf import read_latlondepth
from qcore.pool_wrapper import PoolWrapper

INPUT = 'input'
OUTPUT = 'output'

test_data_save_dir = '/home/jpa198/test_space/im_calc_test/pickled/'
REALISATION = 'PangopangoF29_HYP01-10_S1244'
data_taken = {'horizdist': False,
              'readStationCoordsFile': False,
              'computeSourcetoSiteDistance': False,
              'source_to_distance': False,
              'computeRrup': False,
              }


class Point:
    def __init__(self):
        self.Lat = None
        self.Lon = None
        self.Depth = None
        self.r_rups = None
        self.r_jbs = None

def horizdist(loc1, loc2_lat, loc2_lon):
    """From ComputeSourceToSiteDistance.m """
    # TODO: consider using geopy
    # computes great circle distance between 2 set of (lat, lng) (in degrees)
    function = 'horizdist'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, INPUT, function + '_loc1.P'), 'wb') as save_file:
            pickle.dump(loc1, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, INPUT, function + '_loc2_lat.P'), 'wb') as save_file:
            pickle.dump(loc2_lat, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, INPUT, function + '_loc2_lon.P'), 'wb') as save_file:
            pickle.dump(loc2_lon, save_file)

    EARTH_RADIUS_MEAN = 6371.0072   # Authalic mean radius in km

    # calculations are all in radians
    lat_1, lon_1, lat_2, lon_2 = map(radians, (loc1.Lat, loc1.Lon, loc2_lat, loc2_lon))
    lat = lat_2 - lat_1
    lon = lon_2 - lon_1
    d = sin(lat * 0.5) ** 2 + cos(lat_1) * cos(lat_2) * sin(lon * 0.5) ** 2
    h = 2.0 * EARTH_RADIUS_MEAN * asin(sqrt(d))
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, OUTPUT, function + '_h.P'), 'wb') as save_file:
            pickle.dump(h, save_file)
        data_taken[function] = True
    return h


def readStationCoordsFile(station_file, match_stations=None):
    """Based on readStationCordsFile.m
    """
    function = 'readStationCoordsFile'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, INPUT, function + '_station_file.P'), 'wb') as save_file:
            pickle.dump(station_file, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, INPUT, function + '_match_stations.P'), 'wb') as save_file:
            pickle.dump(match_stations, save_file)
    stations = {}
    get_all_stations = match_stations is None or not any(match_stations)
    with open(station_file, 'r') as fp:
        for line in fp:
            station_info = line.split()
            temp_point = Point()
            station_name = station_info[2]
            # TODO: Graceful error when invalid file read
            if get_all_stations or station_name in match_stations:
                temp_point.Lon = float(station_info[0])
                temp_point.Lat = float(station_info[1])
                temp_point.Depth = float(0)
                stations[station_name] = temp_point
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, OUTPUT, function + '_stations.P'), 'wb') as save_file:
            pickle.dump(stations, save_file)
        data_taken[function] = True
    return stations


def computeSourcetoSiteDistance(finite_fault, Site):
    """ Purpose: compute the distance in km from the finite fault plane to the
    site (of an instrument or other).
    Based on ComputeSourceToSiteDistance.m """

    function = 'computeSourcetoSiteDistance'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, INPUT, function + '_finite_fault.P'), 'wb') as save_file:
            pickle.dump(finite_fault, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, INPUT, function + '_Site.P'), 'wb') as save_file:
            pickle.dump(Site, save_file)

    # start values, no distance should be longer than this
    r_jb = sys.maxsize
    r_rup = sys.maxsize
    r_x = 'X'
    min_depth = sys.maxsize

    # for subfaults, calculate distance, update if shortest
    for fault_i in finite_fault:

        h = horizdist(Site, fault_i['lat'], fault_i['lon'])
        v = Site.Depth - fault_i['depth']

        if abs(h) < r_jb:
            r_jb = h

        d = sqrt(h ** 2 + v ** 2)
        if d < r_rup:
            r_rup = d
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, OUTPUT, function + '_r_rup.P'), 'wb') as save_file:
            pickle.dump(r_rup, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, OUTPUT, function + '_r_jb.P'), 'wb') as save_file:
            pickle.dump(r_jb, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, OUTPUT, function + '_r_x.P'), 'wb') as save_file:
            pickle.dump(r_x, save_file)
        data_taken[function] = True
    return r_rup, r_jb, r_x


def source_to_distance(packaged_data):
    function = 'source_to_distance'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, INPUT, function + '_packaged_data.P'),
                  'wb') as save_file:
            pickle.dump(packaged_data, save_file)

    finite_fault, station, station_name = packaged_data
    dist = computeSourcetoSiteDistance(finite_fault, station)

    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, OUTPUT, function + '_station_name.P'), 'wb') as save_file:
            pickle.dump(station_name, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, OUTPUT, function + '_station.Lat.P'), 'wb') as save_file:
            pickle.dump(station.Lat, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, OUTPUT, function + '_station.Lon.P'), 'wb') as save_file:
            pickle.dump(station.Lon, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, OUTPUT, function + '_dist.P'), 'wb') as save_file:
            pickle.dump(dist, save_file)
        data_taken[function] = True
    return station_name, station.Lat, station.Lon, dist


def computeRrup(station_file, srf_file, match_stations, n_processes):
    """Wrapper function to calculate the rupture distance from the station and srf files"""

    # read in list of stations
    stations = readStationCoordsFile(station_file, match_stations)
    rrup_data = []

    # read in the rupture file
    try:
        finite_fault = read_latlondepth(srf_file)
    except IOError:
        print('SRF filename is not valid. Returning from function computeRrup')
        raise
        return

    function = 'computeRrup'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, INPUT, function + '_station_file.P'),
                  'wb') as save_file:
            pickle.dump(station_file, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, INPUT, function + '_srf_file.P'),
                  'wb') as save_file:
            pickle.dump(srf_file, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, INPUT, function + '_match_stations.P'),
                  'wb') as save_file:
            pickle.dump(match_stations, save_file)
        with open(os.path.join(test_data_save_dir, REALISATION, INPUT, function + '_n_processes.P'),
                  'wb') as save_file:
            pickle.dump(n_processes, save_file)

    # loop over the stations
    # TODO: pass the pool size somehow
    p = PoolWrapper(n_processes)
    packaged_data_list = []
    for station_name, station in stations.iteritems():
        packaged_data_list.append((finite_fault, station, station_name))

    ret_val = p.map(source_to_distance, packaged_data_list)
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, REALISATION, OUTPUT, function + '_ret_val.P'), 'wb') as save_file:
            pickle.dump(ret_val, save_file)
        data_taken[function] = True
    return ret_val
