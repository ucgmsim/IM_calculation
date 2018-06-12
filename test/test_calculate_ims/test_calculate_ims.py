import pytest
from datetime import datetime
import os
import numpy as np
import sys
import getpass
import shutil
import errno
from qcore import shared
import csv

TEST_FOLDER = os.path.abspath(os.path.dirname(__file__))
SCRIPT = os.path.abspath(os.path.join(TEST_FOLDER, '..', '..', 'calculate_ims.py'))
print(SCRIPT)

WHOLE_BENCH_PATH = '/nesi/projects/nesi00213/dev/impp_datasets/darfield_benchmark/im_obs_090.csv'
INPUT_DIR = os.path.join(TEST_FOLDER,'sample1','input')
INPUT_FILES = os.path.join(INPUT_DIR, 'single_files')
BENCHMARK = os.path.join(INPUT_DIR, 'benchmark_obs_090.csv')

OUTPUT_DIR = os.path.join(TEST_FOLDER,'sample1','output')
IDENTIFIER = 'darfield_obs_090'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, IDENTIFIER, IDENTIFIER + '.csv')
OUTPUT_SUBDIR = os.path.join(OUTPUT_DIR, IDENTIFIER, 'stations')

COMP = '090'
STATIONS = ['ASHS', 'BFZ', 'CCC', 'CHHC', 'CMHS', 'DCZ', 'HORC', 'LINC', 'PRPC', 'WEL']
ERROR_LIMIT = 0.01


# def setup_module(scope="module"):
#     """ create a tmp directory for storing output from test"""
#     print "----------setup_module----------"
#     try:
#         os.mkdir(OUTPUT_DIR)
#     except OSError as e:
#         if e.errno != errno.EEXIST:
#             raise
# def write_benchmark_csv(whole_bench_path, sample_bench_path):
#     with open(whole_bench_path, 'r') as whole:
#         buf = whole.readlines()
#
#     with open(sample_bench_path, 'w') as file_writer:
#         file_writer.write(buf[0])
#         for line in buf[1:]:
#             line_seg = line.split(',')
#             if line_seg[0] in STATIONS:
#                 file_writer.write(line)
#
#
# write_benchmark_csv(WHOLE_BENCH_PATH, BENCHMARK)


def get_result_dict(sample_path):
    result_dict = {}
    comp = None
    with open(sample_path, 'r') as result_reader:
        buf = result_reader.readlines()
        measures = buf[0].strip().split(',')[2:]
        for i in range(len(measures)):
            if 'pSA' in measures[i]:
                pre, suf = measures[i].split('_')
                num = '{:.10f}'.format(float(suf.replace('p', '.')))
                measures[i] = pre + '_' + num
        for line in buf[1:]:
            line_seg = line.strip().split(',')
            station = line_seg[0]
            comp = line_seg[1]
            im_values = line_seg[2:]
            result_dict[station] = {}
            for i in range(len(im_values)):
                result_dict[station][measures[i]] = im_values[i]

        return result_dict, comp, measures


def compare_floats(float1, float2, error_limit=ERROR_LIMIT):
    """first test if the two numpy arrays are of the same shape
       if pass, then test if the relative error between the two arrays are <= the preset error limit
       if any of the test fails, assertion error will be raised;
       float1: a float from sample output, will be used as the denominator,
       float2: a float from test output, makes part of the numerator.
       error_limit: preset error_limit to be compared with the relative error (array1-array2)/array1
    """
    relative_error = abs(float1 - float2) / float1
    print relative_error <= error_limit


def test_calculate_ims():
    benchmark_dict, bench_comp, bench_measures = get_result_dict(BENCHMARK)
    output_dict, output_comp, output_measures = get_result_dict(OUTPUT_FILE)

    # assert bench_comp == output_comp
    for station in benchmark_dict.keys():
        for im in benchmark_dict[station].keys():
            try:
                float1 = float(benchmark_dict[station][im])
                float2 = float(output_dict[station][im])
                relative_error = abs(float1 - float2) / float1
                if relative_error > ERROR_LIMIT:
                    print("{}:{} relative error:{} of output {}, benchmark {} exceeding 1%".format(station, im, relative_error, float2, float1))
            except KeyError:
                    print('{} does not exists in output_dict'.format(im))
                    continue
            except ValueError:
                print("Benchmark {} value is None".format(im))
test_calculate_ims()
# output_reader = csv.DictReader(output_file, delimiter=',')
#         output_measures = output_reader.fieldnames

# def test_calculate_ims():
#     """ test calculate_ims.py """
#     print "---------test_calculate_ims------------"
#     cmd = "python {} {} a -o {} -i {} -c {} -e".format(SCRIPT, INPUT_FILES, OUTPUT_DIR, IDENTIFIER, COMP)
#     out, err = shared.exe(cmd)  # the most important function to execute the whole script
#     print(out, err)

# def test_each_im():
#     with open(OUTPUT_FILE, 'r') as output_file:
#         output_reader = csv.DictReader(output_file, delimiter=',')
#         output_measures = output_reader.fieldnames
#
#     with open(BENCHMARK, 'r') as benchmark_file:
#         benchmark_reader = csv.DictReader(benchmark_file, delimiter=',')
#         benchmark_measures = benchmark_reader.fieldnames

#     new_header = ['station, component']
#     for measure in benchmark_measures:
#         if 'pSA' in measure:
#             pre, suf = measure.split('_')
#             num = suf.replace('p','.')[:15]
#             output_measure = pre + '_' + num
#             print(output_measure)
#             measure = output_measure
#         if measure in output_measures:
#             new_header.append(measure)










#
# #
# # def teardown_module():
# #     """ delete the tmp directory if it is empty"""
# #     print "---------teardown_module------------"
# #     if len(os.listdir(OUTPUT_DIR)) == 0:
# #         try:
# #             shutil.rmtree(OUTPUT_DIR)
# #         except (IOError, OSError) as (e):
# #             sys.exit(e)



#
# def test_im_values(station_name):
#     pass