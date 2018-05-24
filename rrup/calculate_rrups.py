
import rrup as rrup
import argparse
import fakedatabase as database

parser = argparse.ArgumentParser('calculate_rrups.py')

parser.add_argument('station_file', type=str)
parser.add_argument('srf_file', type=str)
parser.add_argument('-np', '--processes', type=int, default='4')
parser.add_argument('-s', '--stations', nargs='+', help="space delimited list of stations", default=None)
parser.add_argument('-o', '--output', type=str, default='rrups.csv')

args = parser.parse_args()

db = database.FakeDB('rrups.json')

rrup.computeRrup(db, args.station_file, args.srf_file, args.stations, args.processes)


fname = args.output

with open(fname, 'w') as f:
    f.write("Station_name, r_rup, r_jbs, lon, lat\n")
    sorted_stations = sorted(db.rrups.keys())
    for stat in sorted_stations:
        values = db.rrups[stat]
        r_rups = values['r_rups']
        r_jbs = values['r_jbs']
        lat = values['lat']
        lon = values['lon']
        f.write("%s, %s, %s, %s, %s\n" % (stat, r_rups, r_jbs, lon, lat))
