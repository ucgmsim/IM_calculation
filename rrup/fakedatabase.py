import json
import os

import numpy as np
from database import AbstractDatabase


class FakeDB(AbstractDatabase):
    def __init__(self, output_file):
        self.output_file = output_file
        self.stations = {}
        self.event = {}
        self.rrups = {}
        self.period = []
        self.stations_GMPE_calculated = 0
        self.gmpe = {}
        self.gmpe_psa = {}

    def add_times_to_station(self, name, is_simulation, NT, DT, time_offset):
        pass

    # TODO: this does not work for the moment being but it is still here just in case
    def set_values(self, values_name, values):
        if values_name == "ims":
            self.stations = dict(values)
        elif values_name == "rrups":
            self.rrups = dict(values)
        elif values_name == "gmpe":
            for station in values.keys():
                self.gmpe[station] = values[station]
            self.stations_GMPE_calculated = 1
        elif values_name == "gmpe_psa":
            self.gmpe_psa = dict(values)
        elif values_name == "period":
            self.period = values
        else:
            print "Unknown field", values_name
            print "Not initializing, please "
            exit(1)

    def initialize(self):
        if os.path.isfile(self.output_file):
            print "OutputFile %s exists. Contents will get erased" % self.output_file
            # TODO: eventually allow the json file to be loaded
            # with open(self.output_file) as f:
            #     super_dict = json.load(f, encoding="utf-8")
            #     to_load = {"ims" : self.stations, "rrups" : self.rrups,\
            #                "gmpe" : self.gmpe, "gmpe_psa" : self.gmpe_psa, "period" : self.period}
            #
            #     for section in to_load.keys():
            #         if super_dict.has_key(section):
            #            self.set_values(section, super_dict[section])

        else:
            print "The following file will contain all the data:", self.output_file


    def insert_station(self, is_simulation, name):
        self.stations[name] = {}

    def add_pSA_to_station(self, name, is_simulation, measure, component, value, period):
        self.period = list(period)
        if not self.stations[name].has_key("pSA"):
            self.stations[name]["pSA"] = {component: {}}

        for i in range(len(value)):
            if self.stations[name]["pSA"].has_key(component):
                self.stations[name]["pSA"][component][period[i]] = value[i]
            else:
                self.stations[name]["pSA"][component] = {period[i]: value[i]}

    def add_im_to_station(self, name, is_simulation, measure, component, value):
        try:
            self.stations[name][measure][component] = value
        except KeyError:
            self.stations[name][measure] = {component: value}

    def dump(self):
        super_dict = {
            "ims" : self.stations,
            "rrups" : self.rrups,
            "gmpe" : self.gmpe,
            "gmpe_psa" : self.gmpe_psa,
            "period" : self.period
        }
        with open(self.output_file, "w") as f:
            json.dump(super_dict, f, indent=2, separators=(',', ': '))

    def insert_event(self, run_name, is_simulation, magnitude, rake, dip, Ztor, period):
        self.event["run_name"] = run_name
        self.event["is_simulation"] = is_simulation
        self.event["magnitude"] = magnitude
        self.event["rake"] = rake
        self.event["dip"] = dip
        self.event["Ztor"] = Ztor
        self.event["period"] = period

    def get_calculated(self):
        calculated_ims = 0
        calculated_gmpe = 0
        calculated_rrups = 0
        if len(self.stations) > 0:
            calculated_ims = 1
        if len(self.gmpe) > 0:
            calculated_gmpe = 1
        if len(self.rrups) > 0:
            calculated_rrups = 1
        return calculated_ims, calculated_rrups, 1, 0

    def add_rrups_to_station(self, name, lat, lon, r_rups, r_jbs):
        self.rrups[name] = {"lat": lat, "lon": lon, "r_rups": r_rups, "r_jbs": r_jbs}

    def save(self):
        print "Faking a save() from a SQL DB"

    def get_station_names(self, is_simulation, rrups=False):
        if not rrups:
            return self.stations.keys()
        else:
            return self.rrups.keys()

    def print_db(self):
        print "stations: ", self.stations
        print "rrups", self.rrups
        print "geom", self.geom

    def get_geom(self):
        data_to_return = []
        for station_name in self.stations:
            station = self.stations[station_name]
            for measure in station.keys():
                print "measure", measure
                if measure == 'pSA':
                    x_values = station[measure]['000']
                    y_values = station[measure]['090']
                    assert (len(x_values) == len(y_values))
                    for period in x_values:
                        measure_string = "%f,%f" % (x_values[period], y_values[period])
                        data_to_return.append([station_name, measure, 'geom', measure_string, period])
                else:
                    measure_string = "%f,%f" % (station[measure]['000'], station[measure]['090'])
                    data_to_return.append([station_name, measure, 'geom', measure_string, None])

        return data_to_return

    def insert_multiple_im(self, data):
        for datum in data:
            # unpacking
            name, measure_name, component, value, period = datum
            if measure_name == "pSA":
                try:
                    self.stations[name][measure_name][component]
                except KeyError:
                    self.stations[name][measure_name][component] = {}

                self.stations[name][measure_name][component][period] = value

            else:
                self.stations[name][measure_name][component] = value

    def set_stations_GMPE_calculated(self, value):
        self.stations_GMPE_calculated = value

    def get_allowable_periods(self):
        print "---------->", self.period
        return self.period

    def insert_GMPE(self, im, period, vs30, rrup, im_rscaling, sigma_im_rscaling):
        if im == "pSA":
            if self.gmpe_psa.has_key(rrup):
                self.gmpe_psa[rrup][period] = {"im_rscaling": im_rscaling, "sigma_im_rscaling": sigma_im_rscaling, \
                                               "vs30": vs30}
            else:
                self.gmpe_psa[rrup] = {period: {"im_rscaling": im_rscaling, "sigma_im_rscaling": sigma_im_rscaling, \
                                                "vs30": vs30}}
            return
        # other IMS
        if self.gmpe.has_key(im):
            self.gmpe[im][rrup] = {"im_rscaling": im_rscaling, "sigma_im_rscaling": sigma_im_rscaling, \
                                   "vs30": vs30}
        else:
            self.gmpe[im] = {rrup: {"im_rscaling": im_rscaling, "sigma_im_rscaling": sigma_im_rscaling, \
                                    "vs30": vs30}}

    def get_GMPE_vs30s(self):
        vs30 = []
        selected_im = self.gmpe['PGV']
        for rrup in selected_im.keys():
            vs30.append(selected_im[rrup]["vs30"])
        return list(set(vs30))

    def get_rrups(self, is_simulation, stations):
        sorted_stations = sorted(self.stations.keys())
        return [self.rrups[station]['r_rups'] for station in sorted_stations]

    def get_IM(self, is_simulation, stations, component, measure, period, ratio, empirical=False, include_name=False):
        result = []
        sorted_stations = sorted(self.stations.keys())
        for station in sorted_stations:
            try:
                value = self.stations[station][measure][component]
                if not period is None and measure == "pSA":
                    value = self.stations[station][measure][component][period]
                if include_name:
                    result.append([value, station])
                else:
                    result.append([value])
            except KeyError:
                print "Measure or component not available"
        return result

    def get_stations_with_IM(self, measure, component, is_simulation, stations, time_offsets, include_rrups):
        # 'PGV', component, self.is_simulation, self.stations_to_plot, time_offsets=True, include_rrups=True
        result = []
        return result

    def get_GMPE(self, im, vs30, period):
        result = []
        if im == "pSA":
            sorted_rrups = sorted(self.gmpe_psa.keys())
            print "pSA thingee", period
            for rrup in sorted_rrups:
                periods = self.gmpe_psa[rrup].keys()
                for current_period in periods:
                    if current_period == period:
                        values = self.gmpe_psa[rrup][period]
                        result.append([rrup, values["im_rscaling"],
                                       values["sigma_im_rscaling"][0]])

        else:
            sorted_rrups = sorted(self.gmpe[im].keys())
            for rrup in sorted_rrups:
                values = self.gmpe[im][rrup]
                if values["vs30"] == vs30:
                    result.append([rrup, values["im_rscaling"], values["sigma_im_rscaling"][0]])

        return result

    def get_pSA(self, is_simulation, station, component):
        try:
            psa = []
            psa_values = self.stations[station]["pSA"][component]
            sorted_periods = sorted(psa_values.keys())
            for period in sorted_periods:
                psa.append(self.stations[station]["pSA"][component][period])
        except KeyError:
            print "Error for", station, component
        return psa

    ##########################
    # Unimplemented methods
    ##########################
    def get_pSA_ratio(self, stations, periods, component, empirical=False):
        print "Calling not implemented method get_pSA_ratio"
        pass

    def get_stations_with_IM_for_all_components(self, measure, is_simulation, stations, time_offsets):
        print "Calling not implemented method get_stations_with_IM_for_all_components"
        pass

    def add_vs30_to_station(self, station_name, value):
        print "Calling not implemented method get_vs30_to_station"
        pass

    def get_additional_periods(self, is_simulation):
        print "Calling not implemented method get_additional_periods"
        pass

    def get_GMPE_stations(self, stations):
        print "Calling not implemented method get_GMPE_stations"
        pass

    def set_IMs_calculated(self, value):
        print "Calling not implemented method set_IMs_calculated"
        pass

    def convert_tuple_to_list(self, tuple_list):
        print "Calling not implemented method convert_tuple_to_list"
        pass

    def get_matched_station_names(self, stations, rrups):
        print "Calling not implemented method get_matched_station_names"
        pass

    def get_ratios_calculated(self):
        print "Calling not implemented method get_ratios_calculated"
        pass

    def close(self):
        print "Calling not implemented method close()"
        pass

    def is_GMPE_calculated(self):
        print "Calling not implemented method is_GMPE_calculated"
        pass

    def get_event_names(self):
        print "Calling not implemented method get_event_names"
        pass

    def set_event_id(self, is_simulation, event_id):
        print "Calling not implemented method set_event_id"
        pass

    def get_station_id(self, name, is_simulation):
        print "Calling not implemented method get_station_id"
        pass

    def set_vs30_calculated(self):
        print "Calling not implemented method set_vs30_calculated"
        pass

    def set_ratios_calculated(self, value):
        print "Calling not implemented method set_ratios_calculated"
        pass

    def get_GMPE_event_id(self):
        print "Calling not implemented method get_GMPE_event_id"
        pass

    def get_measures_for_spatial_plot(self, measures, periods, ratio, is_simulation):
        print "Calling not implemented method get_measures_for_spatial_plot"
        pass

    def event_exists(self, run_name, is_simulation):
        print "Calling not implemented method event_exists"
        pass

    def set_GMPE_station_ratio_calculated(self, value):
        print "Calling not implemented method set_GMPE_station_ratio_calculated"
        pass

    def create_ratios(self, stations):
        print "Calling not implemented method create_ratios"
        pass

    def add_empirical_to_station(self, s_id, measure, period, im_rscaling, sigma_im_rscaling):
        print "Calling not implemented method add_empirical_to_station"
        pass

    def execute_ad_hoc_query(self, query):
        print "Calling not implemented method ad_hoc_query"
        pass

    def get_max_PGV(self, is_simulation):
        print "Calling not implemented method get_max_PGV"
        pass

    def set_rrups_calculated(self, value):
        print "Calling not implemented method set_rrups_calculated"
        pass

    def add_all_rrups(self, data):
        print "Calling not implemented method add_all_rrups"
        pass

    def create_obs_emp_ratios(self, stations):
        print "Calling not implemented method create_empirical_ratios"
        pass

    def get_event_id(self, is_simulation):
        print "Calling not implemented method get_event_id"
        pass

    def get_event_ims(self, is_simulation):
        print "Calling not implemented method get_event_ims"
        pass

    def clear_IM(self):
        print "Calling not implemented method clear_IM"
        pass

    def get_max_pSA(self):
        print "Calling not implemented method get_max_pSA"
        pass

    def clear_GMPE(self):
        print "Calling not implemented method clean_GMPE"
        pass

    def set_GMPE_calculated(self, value):
        print "Calling not implemented method set_GMPE_calculated"
        pass
