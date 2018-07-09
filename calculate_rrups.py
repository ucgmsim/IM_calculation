from rrup import rrup as rrup
import argparse

DEFAULT_N_PROCESSES = 4


def write_and_calculate_rrups(station_file, srf_file, stations=None, processes=DEFAULT_N_PROCESSES):
    rrups = rrup.computeRrup(station_file, srf_file, stations, processes)
    fname = args.output
    with open(fname, 'w') as f:
        f.write("Station_name, lon, lat, r_rup, r_jbs, r_x\n")
        for values in rrups:
            name, lat, lon, (r_rups, r_jbs, r_x) = values
            f.write("%s,%s,%s,%s,%s,%s\n" % (name, lon, lat, r_rups, r_jbs, r_x))


# /nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p6/Runs/Kelly/fd_rt01-h0.400.ll
# /scale_akl_nobackup/filesets/transit/nesi00213/StationInfo/non_uniform_whole_nz_with_real_stations-hh400_v18p6.ll
def get_fd_stations(fd_ll):
    """
    :param fd_ll: path to fd_ll station file
    :return: a list of useful stations
    """
    stations = []
    with open(fd_ll, 'r') as fd:
        lines = fd.readlines()
        for line in lines:
            station = line.strip().split()[-1]
            stations.append(station)
    print("stations",len(stations))
    return stations


if __name__ == "__main__":
    parser = argparse.ArgumentParser('calculate_rrups.py')
    parser.add_argument('fd_file', type=str)
    parser.add_argument('station_file', type=str)
    parser.add_argument('srf_file', type=str)
    parser.add_argument('-np', '--processes', type=int, default=DEFAULT_N_PROCESSES)
    # parser.add_argument('-s', '--stations', nargs='+', help="space delimited list of stations", default=None)
    parser.add_argument('-o', '--output', type=str, default='rrups.csv')

    args = parser.parse_args()
    match_stations = get_fd_stations(args.fd_file)
    write_and_calculate_rrups(args.station_file, args.srf_file, match_stations, args.processes)
