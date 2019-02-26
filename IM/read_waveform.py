import os
import sys

import numpy as np

G = 981


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


def read_ascii_file(f_000, f_090, f_ver, wave_type=None):
    waveform = Waveform()
    waveform.wave_type = wave_type
    waveform.file_type = "EMOD3D_ascii"

    (
        waveform.station_name,
        waveform.NT,
        waveform.DT,
        waveform.time_offset,
    ) = read_ascii_header(f_000)
    skip_header(f_090)
    skip_header(f_ver)

    waveform.times = calculate_timesteps(waveform.NT, waveform.DT)

    values = np.zeros((waveform.NT, 3))

    i = 0
    files = zip(f_000, f_090, f_ver)
    for l_000, l_090, l_ver in files:
        a = [l_090.split(), l_000.split(), l_ver.split()]
        line_values = np.array(a, np.float).transpose()
        n_vals = len(line_values)
        values[i : i + n_vals] = line_values
        i += n_vals

    waveform.values = values

    return waveform


def read_ascii_header(fid):
    # first line of the header (station_name, component)
    header1 = fid.next().split()
    station_name = header1[0]

    # second line of the header
    header2 = fid.next().split()
    NT = np.int(header2[0])
    DT = np.float(header2[1])

    hour = np.float(header2[2])
    minutes = np.float(header2[3])
    seconds = np.float(header2[4])

    time_offset = hour * 60.0 ** 2 + minutes * 60.0 + seconds

    return station_name, NT, DT, time_offset


def skip_header(fid):
    next(fid)
    next(fid)


def calculate_timesteps(NT, DT):
    return np.arange(NT) * DT


def create_waveform_from_data(
    data, wave_type=None, base_waveform=None, NT=None, DT=None, offset=None, name=None
):
    if base_waveform is not None:
        NT = base_waveform.NT
        DT = base_waveform.DT
        offset = base_waveform.time_offset
        name = base_waveform.station_name
        if wave_type is None:
            wave_type = base_waveform.wave_type
    times = calculate_timesteps(NT, DT)
    waveform = Waveform(
        NT=NT,
        DT=DT,
        time_offset=offset,
        wave_type=wave_type,
        values=data,
        file_type="raw_data",
        times=times,
        station_name=name,
    )
    return waveform


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

    print(units)
    if file_type == "ascii":
        return read_ascii_folder(path, station_names, units=units)
    elif file_type == "binary":
        return read_binary_file(
            bbseis,
            comp,
            station_names,
            wave_type=wave_type,
            file_type="binary",
            units=units,
        )
    else:
        print("Could not determine filetype %s" % path)
        return None


def get_station_name_from_filepath(path):
    base_station_name = os.path.basename(path)
    station_name = os.path.splitext(base_station_name)[0]
    return station_name


def read_ascii_folder(path, station_names, units="g"):
    waveforms = list()

    for station in station_names:
        filename_000 = os.path.join(path, station + ".000")
        filename_090 = os.path.join(path, station + ".090")
        filename_ver = os.path.join(path, station + ".ver")
        try:
            f_000 = open(filename_000)
            f_090 = open(filename_090)
            f_ver = open(filename_ver)
        except IOError:
            print(
                "Could not open file %s Ignoring this station"
                % os.path.join(path, station)
            )
            return None, None

        waveform = read_ascii_file(f_000, f_090, f_ver, "acceleration")
        if units == "cm/s^2":
            waveform.values = waveform.values / G
        waveforms.append((waveform, None))
        f_000.close()
        f_090.close()
        f_ver.close()

    return waveforms


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
    return waveforms
