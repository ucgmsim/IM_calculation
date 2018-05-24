import sys
import os
import numpy as np
import im_calculations

sys.path.append('../qcore/qcore/')
from timeseries import BBSeis

G = 981.0
MEASURES  = ['AI', 'CAV', 'Ds575', 'Ds595', 'PGA', 'PGV', 'pSA', 'MMI']
EXTENSIONS = ["000", "090", "ver"]
PSA_PARAMS = {
    "M": 1.0,
    "beta": 0.25,
    "gamma": 0.5,
    "c": 0.05,
    "deltat": 0.005,
    "extented_period": np.logspace(start=np.log10(0.01), stop=np.log10(10.), num=100, base=10),
    "karim20_period":[0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.75,1.0,2.0,3.0,4.0,5.0,7.5,10.0]
}


class Waveform:
    def __init__(self, NT=None, DT=None, time_offset=None, values=None, wave_type=None, file_type=None, times=None,
                 station_name=None):
        self.NT = NT
        self.DT = DT
        self.times = times
        self.values = values
        self.time_offset = time_offset
        self.wave_type = wave_type
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


def read_file(filename, wave_type=None, file_type=None):
    try:
        fid = open(filename)
    except IOError:
        print 'Could not open file %s Ignoring this station' % filename
        return None

    extension = os.path.splitext(filename)[-1]
    if file_type == 'standard' or extension in ['.000', '.090', '.ver']:
        return read_standard_file(fid, wave_type, 'standard')
    else:
        print "Could not determine filetype %s Ignoring this station" % filename
        return None



def read_station_from_binary(input_path,station_name=None):
    bbseries = BBSeis(input_path)

    if not station_name:
        return bbseries.stations
    else:
        try:
            station_index = bbseries.stat_idx[station_name]
        except KeyError:
            sys.exit("staiton name {} does not exist".format(station_name))
        return bbseries.stations[station_index]


# def compute_measures(input_path, station_name=None, ims=MEASURES, component=Ellipsis, period=20, extended_period=False,meta_data=None, output='.'):
#     if isinstance(ims, str):
#         ims = ims.split()   # if only one measure is provided, make it into a list
#
#     for im in ims:














if __name__ == '__main__':
    print(read_station_from_binary('../BB.bin',station_name='112A'))


