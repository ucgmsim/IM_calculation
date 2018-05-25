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
import pool_wrapper

from qcore.srf import read_latlondepth


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
    EARTH_RADIUS_MEAN = 6371.0072   # Authalic mean radius in km

    # calculations are all in radians
    lat_1, lon_1, lat_2, lon_2 = map(radians, (loc1.Lat, loc1.Lon, loc2_lat, loc2_lon))
    lat = lat_2 - lat_1
    lon = lon_2 - lon_1
    d = sin(lat * 0.5) ** 2 + cos(lat_1) * cos(lat_2) * sin(lon * 0.5) ** 2
    h = 2.0 * EARTH_RADIUS_MEAN * asin(sqrt(d))
    return h


def readStationCoordsFile(station_file, match_stations=None):
    """Based on readStationCordsFile.m
    """
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
    return stations


def computeSourcetoSiteDistance(finite_fault, Site):
    """ Purpose: compute the distance in km from the finite fault plane to the
    site (of an instrument or other).
    Based on ComputeSourceToSiteDistance.m """

    # start values, no distance should be longer than this
    r_jb = sys.maxsize
    r_rup = sys.maxsize
    r_x = sys.maxsize
    min_depth = sys.maxsize

    # for subfaults, calculate distance, update if shortest
    for fault_i in finite_fault:

        h = horizdist(Site, fault_i['lat'], fault_i['lon'])
        v = Site.Depth - fault_i['depth']

        if v < min_depth:
            min_depth = v
            r_x = h

        if abs(h) < r_jb:
            r_jb = h

        d = sqrt(h ** 2 + v ** 2)
        if d < r_rup:
            r_rup = d

    return r_rup, r_jb, r_x


def source_to_distance(packaged_data):
    finite_fault, station, station_name = packaged_data
    return station_name, station.Lat, station.Lon, computeSourcetoSiteDistance(finite_fault, station)


def computeRrup(station_file, srf_file, match_stations, n_processes):
    """Wrapper function to calculate the rupture distance from the station and srf files"""

    # read in list of stations
    stations = readStationCoordsFile(station_file, match_stations)
    rrup_data = []

    # read in the rupture file
    try:
        finite_fault = read_latlondepth(srf_file)
    except IOError:
        print 'SRF filename is not valid. Returning from function computeRrup'
        raise
        return

    # loop over the stations
    # TODO: pass the pool size somehow
    p = pool_wrapper.PoolWrapper(n_processes)
    packaged_data_list = []
    for station_name, station in stations.iteritems():
        packaged_data_list.append((finite_fault, station, station_name))

    return p.map(source_to_distance, packaged_data_list)
