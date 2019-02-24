import os

import pickle

from test.test_common_set_up import set_up, INPUT, OUTPUT, compare_dicts
import rrup.rrup as rrup


class TestPickleTesting():
    def test_horizdist(self, set_up):
        function = 'horizdist'
        for root_path in set_up:
            with open(os.path.join(root_path, INPUT, function + '_loc1.P'), 'rb') as load_file:
                loc1 = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_loc2_lat.P'), 'rb') as load_file:
                loc2_lat = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_loc2_lon.P'), 'rb') as load_file:
                loc2_lon = pickle.load(load_file)

            actual_h = rrup.horizdist(loc1, loc2_lat, loc2_lon)

            with open(os.path.join(root_path, OUTPUT, function + '_h.P'), 'rb') as load_file:
                expected_h = pickle.load(load_file)

            assert expected_h == actual_h

    def test_readStationCoordsFile(self, set_up):
        function = 'readStationCoordsFile'
        for root_path in set_up:
            with open(os.path.join(root_path, INPUT, function + '_match_stations.P'), 'rb') as load_file:
                match_stations = pickle.load(load_file)

            station_file = os.path.join(root_path, INPUT, 'sample_station.ll')

            actual_stations = rrup.readStationCoordsFile(station_file, match_stations)

            with open(os.path.join(root_path, OUTPUT, function + '_stations.P'), 'rb') as load_file:
                expected_stations = pickle.load(load_file)

            compare_dicts(expected_stations, actual_stations)

    def test_computeSourcetoSiteDistance(self, set_up):
        function = 'computeSourcetoSiteDistance'
        for root_path in set_up:
            with open(os.path.join(root_path, INPUT, function + '_finite_fault.P'), 'rb') as load_file:
                finite_fault = pickle.load(load_file)
            with open(os.path.join(root_path, INPUT, function + '_Site.P'), 'rb') as load_file:
                Site = pickle.load(load_file)

            actual_r_rup, actual_r_jb, actual_r_x = rrup.computeSourcetoSiteDistance(finite_fault, Site)

            with open(os.path.join(root_path, OUTPUT, function + '_r_rup.P'), 'rb') as load_file:
                expected_r_rup = pickle.load(load_file)
            with open(os.path.join(root_path, OUTPUT, function + '_r_jb.P'), 'rb') as load_file:
                expected_r_jb = pickle.load(load_file)
            with open(os.path.join(root_path, OUTPUT, function + '_r_x.P'), 'rb') as load_file:
                expected_r_x = pickle.load(load_file)

            assert actual_r_rup == expected_r_rup
            assert actual_r_jb == expected_r_jb
            assert actual_r_x == expected_r_x

    def test_source_to_distance(self, set_up):
        function = 'source_to_distance'
        for root_path in set_up:
            with open(os.path.join(root_path, INPUT, function + '_packaged_data.P'), 'rb') as load_file:
                packaged_data = pickle.load(load_file)

            actual_station_name, actual_station_Lat, actual_station_Lon, actual_dist = rrup.source_to_distance(packaged_data)

            with open(os.path.join(root_path, OUTPUT, function + '_station_name.P'), 'rb') as load_file:
                expected_station_name = pickle.load(load_file)
            with open(os.path.join(root_path, OUTPUT, function + '_station.Lat.P'), 'rb') as load_file:
                expected_station_Lat = pickle.load(load_file)
            with open(os.path.join(root_path, OUTPUT, function + '_station.Lon.P'), 'rb') as load_file:
                expected_station_Lon = pickle.load(load_file)
            with open(os.path.join(root_path, OUTPUT, function + '_dist.P'), 'rb') as load_file:
                expected_dist = pickle.load(load_file)

            assert actual_station_name == expected_station_name
            assert actual_station_Lat == expected_station_Lat
            assert actual_station_Lon == expected_station_Lon
            assert actual_dist == expected_dist

    def test_computeRrup(self, set_up):
        function = 'computeRrup'
        for root_path in set_up:
            with open(os.path.join(root_path, INPUT, function + '_match_stations.P'), 'rb') as load_file:
                match_stations = pickle.load(load_file)

            n_processes = 1
            srf_file = os.path.join(root_path, INPUT, 'PangopangoF29_HYP01-10_S1244.srf')
            station_file = os.path.join(root_path, INPUT, 'sample_station.ll')

            actual_ret_val = rrup.computeRrup(station_file, srf_file, match_stations, n_processes)

            with open(os.path.join(root_path, OUTPUT, function + '_ret_val.P'), 'rb') as load_file:
                expected_ret_val = pickle.load(load_file)

            assert expected_ret_val == actual_ret_val
