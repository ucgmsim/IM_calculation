
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser('calculate_rrups.py')

    parser.add_argument('station_file', type=str)
    parser.add_argument('srf_file', type=str)
    parser.add_argument('-np', '--processes', type=int, default=DEFAULT_N_PROCESSES)
    parser.add_argument('-s', '--stations', nargs='+', help="space delimited list of stations", default=None)
    parser.add_argument('-o', '--output', type=str, default='rrups.csv')

    args = parser.parse_args()
    write_and_calculate_rrups(args.station_file, args.srf_file, args.stations, args.processes)
