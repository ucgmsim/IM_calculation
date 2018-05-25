
import rrup as rrup
import argparse

parser = argparse.ArgumentParser('calculate_rrups.py')

parser.add_argument('station_file', type=str)
parser.add_argument('srf_file', type=str)
parser.add_argument('-np', '--processes', type=int, default='4')
parser.add_argument('-s', '--stations', nargs='+', help="space delimited list of stations", default=None)
parser.add_argument('-o', '--output', type=str, default='rrups.csv')

args = parser.parse_args()

rrups = rrup.computeRrup(args.station_file, args.srf_file, args.stations, args.processes)


fname = args.output

with open(fname, 'w') as f:
    f.write("Station_name, lon, lat, r_rup, r_jbs, r_x\n")
    for values in rrups:
        name, lat, lon, (r_rups, r_jbs, r_x) = values
        f.write("%s, %s, %s, %s, %s, %s\n" % (name, lon, lat, r_rups, r_jbs, r_x))
