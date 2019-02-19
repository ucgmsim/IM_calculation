import os
import numpy as np
import glob
import itertools
import sys
import pickle

g = 981

test_data_save_dir = '/home/jpa198/test_space/im_calc_test/pickled/Hossack_HYP01-10_S1244'
REALISATION = 'Hossack_HYP01-10_S1244'
data_taken = {'calculate_timesteps': False,
              'read_waveforms': False,
              'read_one_station_from_bbseries': False,
              'read_binary_file': False,
              }


class Waveform:
    def __init__(
        self,
        NT=None,
        DT=None,
        time_offset=None,
        values=None,
        wave_type=None,
        file_type=None,
        times=None,
        station_name=None,
    ):
        self.NT = NT  # number of entries  how many data points on the plot
        self.DT = DT  # time step
        self.times = times  # array of x values
        self.values = values  # v
        self.time_offset = time_offset
        self.wave_type = wave_type  # v or a
        self.file_type = file_type
        self.station_name = station_name


def calculate_timesteps(NT, DT):
    function = 'calculate_timesteps'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, function + '_NT.P'), 'wb') as save_file:
            pickle.dump(NT, save_file)
        with open(os.path.join(test_data_save_dir, function + '_DT.P'), 'wb') as save_file:
            pickle.dump(DT, save_file)
    ret_val = np.arange(NT) * DT
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, function + '_ret_val.P'), 'wb') as save_file:
            pickle.dump(ret_val, save_file)
        data_taken[function] = True
    return ret_val


def read_waveforms(
    path,
    bbseis,
    station_names=None,
    comp=Ellipsis,
    wave_type=None,
    file_type=None,
    units="g",
):
    """
    read either a ascii or binary file
    :param path:
    :param station_names:
    :param comp:
    :param wave_type:
    :param file_type:
    :return: a list of waveforms
    """
    function = 'read_waveforms'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, function + '_path.P'), 'wb') as save_file:
            pickle.dump(path, save_file)
        with open(os.path.join(test_data_save_dir, function + '_bbseis.P'), 'wb') as save_file:
            pickle.dump(bbseis, save_file)
        with open(os.path.join(test_data_save_dir, function + '_station_names.P'), 'wb') as save_file:
            pickle.dump(station_names, save_file)
        with open(os.path.join(test_data_save_dir, function + '_comp.P'), 'wb') as save_file:
            pickle.dump(comp, save_file)
        with open(os.path.join(test_data_save_dir, function + 'wave_type.P'), 'wb') as save_file:
            pickle.dump(wave_type, save_file)
        with open(os.path.join(test_data_save_dir, function + '_file_type.P'), 'wb') as save_file:
            pickle.dump(file_type, save_file)
        with open(os.path.join(test_data_save_dir, function + '_units.P'), 'wb') as save_file:
            pickle.dump(units, save_file)

    print(units)
    if file_type == "binary":
        ret_val = read_binary_file(
            bbseis,
            comp,
            station_names,
            wave_type=wave_type,
            file_type="binary",
            units=units,
        )

        if not data_taken[function]:
            with open(os.path.join(test_data_save_dir, function + '_ret_val.P'), 'wb') as save_file:
                pickle.dump(ret_val, save_file)
            data_taken[function] = True

        return ret_val
    else:
        print("Could not determine filetype %s" % path)
        if not data_taken[function]:
            with open(os.path.join(test_data_save_dir, function + '_ret_val.P'), 'wb') as save_file:
                pickle.dump(None, save_file)
            data_taken[function] = True
        return None


def read_one_station_from_bbseries(
    bbseries, station_name, comp, wave_type=None, file_type=None
):
    """
    read one station data into a waveform obj
    :param bbseries:
    :param station_name:
    :param comp:
    :param wave_type:
    :param file_type:
    :return: a waveform obj with either acc or vel in values
    """
    function = 'read_one_station_from_bbseries'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, function + '_bbseries.P'), 'wb') as save_file:
            pickle.dump(bbseries, save_file)
        with open(os.path.join(test_data_save_dir, function + '_station_name.P'), 'wb') as save_file:
            pickle.dump(station_name, save_file)
        with open(os.path.join(test_data_save_dir, function + '_comp.P'), 'wb') as save_file:
            pickle.dump(comp, save_file)
        with open(os.path.join(test_data_save_dir, function + '_wave_type.P'), 'wb') as save_file:
            pickle.dump(wave_type, save_file)
        with open(os.path.join(test_data_save_dir, function + '_file_type.P'), 'wb') as save_file:
            pickle.dump(file_type, save_file)

    waveform = Waveform()  # instance of Waveform
    waveform.wave_type = wave_type
    waveform.file_type = file_type
    waveform.station_name = station_name
    waveform.NT = bbseries.nt  # number of timesteps
    waveform.DT = bbseries.dt  # time step
    waveform.time_offset = bbseries.start_sec  # time offset
    waveform.times = calculate_timesteps(
        waveform.NT, waveform.DT
    )  # array of time values

    try:
        if wave_type == "a":
            waveform.values = bbseries.acc(
                station=station_name, comp=comp
            )  # get timeseries/acc for a station
        elif wave_type == "v":
            waveform.values = bbseries.vel(station=station_name, comp=comp)
    except KeyError:
        sys.exit("station name {} does not exist".format(station_name))
    function = 'read_waveforms'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, function + '_waveform.P'), 'wb') as save_file:
            pickle.dump(waveform, save_file)
        data_taken[function] = True
    return waveform


def read_binary_file(
    bbseries, comp, station_names=None, wave_type=None, file_type=None, units="g"
):
    """
    read all stations into a list of waveforms
    :param input_path:
    :param comp:
    :param station_names:
    :param wave_type:
    :param file_type:
    :return: [(waveform_acc, waveform_vel])
    """
    function = 'read_binary_file'
    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, function + '_bbseries.P'), 'wb') as save_file:
            pickle.dump(bbseries, save_file)
        with open(os.path.join(test_data_save_dir, function + '_station_names.P'), 'wb') as save_file:
            pickle.dump(station_names, save_file)
        with open(os.path.join(test_data_save_dir, function + '_comp.P'), 'wb') as save_file:
            pickle.dump(comp, save_file)
        with open(os.path.join(test_data_save_dir, function + '_wave_type.P'), 'wb') as save_file:
            pickle.dump(wave_type, save_file)
        with open(os.path.join(test_data_save_dir, function + '_file_type.P'), 'wb') as save_file:
            pickle.dump(file_type, save_file)
        with open(os.path.join(test_data_save_dir, function + '_units.P'), 'wb') as save_file:
            pickle.dump(units, save_file)

    waveforms = []
    # if not station_names:
    #     station_names = bbseries.stations.name
    for station_name in station_names:
        waveform_acc = read_one_station_from_bbseries(
            bbseries, station_name, comp, wave_type="a", file_type=file_type
        )  # TODO should create either a or v not both
        waveform_vel = read_one_station_from_bbseries(
            bbseries, station_name, comp, wave_type="v", file_type=file_type
        )
        if units == "cm/s^2":
            waveform_acc.values = waveform_acc.values / 981
        waveforms.append((waveform_acc, waveform_vel))

    if not data_taken[function]:
        with open(os.path.join(test_data_save_dir, function + '_waveforms.P'), 'wb') as save_file:
            pickle.dump(waveforms, save_file)
        data_taken[function] = True
    return waveforms
