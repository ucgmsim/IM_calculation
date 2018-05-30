#TODO ADD velocities attribute calculation for standard file waveform obj
import sys
import os
import numpy as np
import im_calculations

from qcore import timeseries

G = 981.0
MEASURES  = ['AI', 'CAV', 'Ds575', 'Ds595', 'PGA', 'PGV', 'pSA', 'MMI']
EXTENSIONS = ["000", "090", "ver"]


class Waveform:
    def __init__(self, NT=None, DT=None, time_offset=None, values=None, wave_type=None, file_type=None, times=None,
                 station_name=None):
        self.NT = NT  # number of entries  how many data points on the plot
        self.DT = DT  # time step
        self.times = times  # array of x values
        self.values = values  # v
        self.time_offset = time_offset
        self.wave_type = wave_type  # v or a
        self.file_type = file_type
        self.station_name = station_name


def read_standard_file(fid, wave_type=None, file_type=None):
    waveform = Waveform()
    waveform.wave_type = wave_type
    waveform.file_type = file_type

    # first line of the header (station_name, component)
    header1 = fid.next().split()
    waveform.station_name = header1[0]

    # second line of the header
    header2 = fid.next().split()
    waveform.NT = np.int(header2[0])
    waveform.DT = np.float(header2[1])
    values = np.zeros(waveform.NT)

    hour = np.float(header2[2])
    minutes = np.float(header2[3])
    seconds = np.float(header2[4])
    waveform.time_offset = hour * 60. ** 2 + minutes * 60. + seconds

    waveform.times = calculate_timesteps(waveform.NT, waveform.DT)

    i = 0
    for line in fid:
        line_values = map(np.float, line.split())
        n_vals = len(line_values)
        values[i:i + n_vals] = line_values
        i += n_vals

    # close the file
    fid.close()
    waveform.values = values

    return waveform


def calculate_timesteps(NT, DT):
    return np.arange(NT) * DT


def create_waveform_from_data(data, wave_type=None, base_waveform=None, NT=None, DT=None, offset=None, name=None):
    if base_waveform is not None:
        NT = base_waveform.NT
        DT = base_waveform.DT
        offset = base_waveform.time_offset
        name = base_waveform.station_name
        if wave_type is None:
            wave_type = base_waveform.wave_type
    times = calculate_timesteps(NT, DT)
    waveform = Waveform(NT=NT, DT=DT, time_offset=offset, wave_type=wave_type, values=data, file_type='raw_data',
                        times=times, station_name=name)
    return waveform


def read_file(filename, station_name=None, comp=Ellipsis, wave_type=None, file_type=None):
    try:
        fid = open(filename)
    except IOError:
        print 'Could not open file %s Ignoring this station' % filename
        return None

    extension = os.path.splitext(filename)[-1]
    if file_type == 'standard' or extension in ['.000', '.090', '.ver']:
        return read_standard_file(fid, wave_type, 'standard')
    elif file_type == 'binary':
        return read_binary_file(filename, comp, station_name, wave_type=wave_type, file_type='binary')
    else:
        print "Could not determine filetype %s Ignoring this station" % filename
        return None


def read_one_station_from_bbseries(bbseries, station_name, comp, wave_type=None, file_type=None):
    waveform = Waveform()  # instance of Waveform
    waveform.wave_type = wave_type
    waveform.file_type = file_type
    waveform.station_name = station_name
    waveform.NT = bbseries.nt   # number of timesteps
    waveform.DT = bbseries.dt   # time step
    waveform.time_offset = bbseries.start_sec   # time offset
    waveform.times = calculate_timesteps(waveform.NT, waveform.DT)  # array of time values

    # print("nunber of timesteps",waveform.NT)
    # print("time step", waveform.DT)
    # print("duration", bbseries.duration)
    # print("stations", bbseries.stations)

    try:
        print("start values",station_name,comp)
        waveform.values = bbseries.acc(station=station_name, comp=comp)  # get timeseries/acc for a station
        print(waveform.values)
    except KeyError:
        sys.exit("staiton name {} does not exist".format(station_name))
    return waveform


def read_binary_file(input_path, comp, station_name=None, wave_type=None, file_type=None):
    bbseries = timeseries.BBSeis(input_path)

    if station_name:
        waveform = read_one_station_from_bbseries(bbseries, station_name, comp, wave_type=wave_type, file_type=file_type)
        return waveform
    else:
        waveforms = []
        station_names = bbseries.stations.name
        for station_name in station_names:
            waveform = read_one_station_from_bbseries(bbseries, station_name, comp, wave_type=wave_type, file_type=file_type)
            waveforms.append(waveform)
        return waveforms


if __name__ == '__main__':
    print(read_file('../BB.bin', station_name='112A', file_type='binary'))


