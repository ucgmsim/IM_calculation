#!/usr/bin/env python2

from argparse import ArgumentParser
from glob import glob
import os

import numpy as np

parser = ArgumentParser()
parser.add_argument('runs_dir', help = 'location to Runs folder')
args = parser.parse_args()

# csv_files here only used to find valid directories (faults)
csv_files = glob(os.path.join(args.runs_dir, '*', 'IM_calc', '*', '*.csv'))
im_faults = set(im_csv.split(os.sep)[-4] for im_csv in csv_files)
del csv_files

for fault in im_faults:
    # prepare input files and output dir
    csv_set = glob(os.path.join(args.runs_dir, fault, 'IM_calc', '*', '*.csv'))
    out_dir = os.path.join(args.runs_dir, fault, 'IM_agg')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # within the set, expect the same columns to be available
    with open(csv_set[0], 'r') as f:
        columns = f.readline().strip().split(',')
    cols = []
    names = []
    for i, c in enumerate(columns):
        # filter out pSA that aren't round numbers, duplicates
        if not (c.startswith('pSA_') and len(c) > 12 or c in names):
            cols.append(i)
            names.append(c)
    cols = tuple(cols)
    dtype = [(n, 'f') for n in names]
    # first 2 columns are actually strings
    dtype[0] = ('station', '|S7')
    dtype[1] = ('component', '|S4')
    dtype = np.dtype(dtype)

    # load all at once, assuming enough memory
    csv_np = []
    header = ['station']
    for csv in csv_set:
        d = np.loadtxt(csv, dtype = dtype, delimiter = ',', \
                       skiprows = 1, usecols = cols)
        csv_np.append(d[d['component'] == 'geom'])
        header.append(os.path.splitext(os.path.basename(csv))[0])
    n_csv = len(csv_np)
    n_stat = csv_np[0].size
    for i in xrange(n_csv - 1):
        assert(np.array_equiv(csv_np[i]['station'], csv_np[i + 1]['station']))

    # store outputs
    dtype = np.dtype(','.join(['|S7'] + ['f'] * n_csv))
    fmt = ','.join(['%s'] + ['%f'] * n_csv)
    header = ','.join(header)
    for c in names[2:]:
        out_file = os.path.join(out_dir, '%s.csv' % (c))
        out_data = np.zeros(n_stat, dtype = dtype)
        out_data['f0'] = csv_np[0]['station']
        for i in xrange(n_csv):
            out_data['f%d' % (i + 1)] = csv_np[i][c]
        np.savetxt(out_file, out_data, fmt = fmt, delimiter = ',', \
                    header = header, comments = '')
