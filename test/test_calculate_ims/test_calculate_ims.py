"""
Comnpars output csvs from calculate_ims.py with the benchmark csvs for selected stations
pytest -v -s test_calculate_ims.py
pytest --cov --cov-report=html -s test_calculate_ims.py
"""

import os
import shutil
from qcore import shared, utils

TEST_FOLDER = os.path.abspath(os.path.dirname(__file__))
SCRIPT = os.path.abspath(os.path.join(TEST_FOLDER, '..', '..', 'calculate_ims.py'))
print(SCRIPT)

INPUT_DIR = os.path.join(TEST_FOLDER,'sample1','input')
INPUT_BINARY = '/home/yzh231/BB_with_siteamp.bin'
INPUT_ASCII = os.path.join(INPUT_DIR, 'single_files')
BENCHMARK = os.path.join(INPUT_DIR, 'new_im_sim_benchmark.csv')

OUTPUT_DIR = os.path.join(TEST_FOLDER, 'sample1', 'output')

IDENTIFIER_BINARY = 'binary_darfield_im_sim'
OUTPUT_BINARY_DIR = os.path.join(OUTPUT_DIR, IDENTIFIER_BINARY)
OUTPUT_BINARY_FILE = os.path.join(OUTPUT_BINARY_DIR, IDENTIFIER_BINARY + '.csv')
OUTPUT_BINARY_META = os.path.join(OUTPUT_BINARY_DIR, IDENTIFIER_BINARY + '_imcalc.info')
OUTPUT_BINARY_SUBDIR = os.path.join(OUTPUT_DIR, IDENTIFIER_BINARY, 'stations')

IDENTIFIER_ASCII = 'ascii_darfield_im_sim'
OUTPUT_ASCII_DIR = os.path.join(OUTPUT_DIR, IDENTIFIER_ASCII)
OUTPUT_ASCII_FILE = os.path.join(OUTPUT_ASCII_DIR, IDENTIFIER_ASCII + '.csv')
OUTPUT_ASCII_META = os.path.join(OUTPUT_ASCII_DIR, IDENTIFIER_ASCII + '_imcalc.info')
OUTPUT_ASCII_SUBDIR = os.path.join(OUTPUT_DIR, IDENTIFIER_ASCII, 'stations')

STATIONS = '2002199 GRY 00020d3 UNK CASH CFW DLX LSRC EWZ PEAA'
PERIODS = '0.01 0.2 0.5 1.0 3.0 4.0 10.0'
COMP_DICT = {'090': '90', 'geom': 'geom', '000': '0', 'ver': 'ver'}
ERROR_LIMIT = 0.01


def setup_module(scope="module"):
    """ create a tmp directory for storing output from test"""
    print "----------setup_module----------"
    utils.setup_dir(OUTPUT_DIR)


def teardown_module():
    """ delete the tmp directory if it is empty"""
    print "---------teardown_module------------"
    for f in os.listdir(OUTPUT_DIR):
        f_path = os.path.join(OUTPUT_DIR, f)
        if len(os.listdir(f_path)) == 0:
            remove_folder(f_path)

    if len(os.listdir(OUTPUT_DIR)) == 0:
        remove_folder(OUTPUT_DIR)


def run_script_calculate_ims(input_path, input_type, identifier):
    """
    :param input_path: INPUT_BINARY/ASCII
    :param input_type: 'b'/'a'
    :param identifier: IDENTIFIER_BINARY/ASCII
    :return: error string
    """
    cmd = 'python {} {} {} -o {} -i {} -t s -n {} -p {}'.format(SCRIPT, input_path, input_type, OUTPUT_DIR, identifier, STATIONS, PERIODS)
    _, err = shared.exe(cmd)
    return err


def test_binary_script_calculate_ims():
    err = run_script_calculate_ims(INPUT_BINARY, 'b', IDENTIFIER_BINARY)
    assert err == ''


def test_ascii_script_calculate_ims():
    err = run_script_calculate_ims(INPUT_ASCII, 'a', IDENTIFIER_ASCII)
    assert err == ''


def get_result_dict(sample_path):
    """
    :param sample_path: path to BENCHMARK
    :return: dict {station: {comp: measure: im_value}}}
    """
    result_dict = {}
    with open(sample_path, 'r') as result_reader:
        buf = result_reader.readlines()
        measures = buf[0].strip().split(',')[2:]
        for i in range(len(measures)):
            if 'pSA' in measures[i]:
                pre, suf = measures[i].split('_')
                num = '{:.9f}'.format(float(suf.replace('p', '.')))
                measures[i] = pre + '_' + num
        for line in buf[1:]:
            line_seg = line.strip().split(',')
            station = line_seg[0]
            comp = line_seg[1]
            im_values = line_seg[2:]
            try:
                result_dict[station].update({comp: {}})
            except KeyError:
                result_dict[station] = {comp: {}}
            for i in range(len(im_values)):
                if measures[i] != '':
                    result_dict[station][comp][measures[i]] = im_values[i]
    return result_dict


def run_test_calculate_ims(test_output_path):
    """
    :param test_output_path: OUTPUT_BINARY/ASCII_FILE
    :return: error string
    """
    benchmark_dict = get_result_dict(BENCHMARK)
    output_dict = get_result_dict(test_output_path)
    passed = 0
    failed = 0
    errors = ''
    warning = ''
    for station in output_dict.keys():
        for comp in output_dict[station].keys():
            for im in output_dict[station][comp].keys():
                try:
                    float1 = float(benchmark_dict[station][COMP_DICT[comp]][im])
                    float2 = float(output_dict[station][comp][im])
                    relative_error = abs(float1 - float2) / float1
                    if relative_error > ERROR_LIMIT:
                        e = "{}: {}: {} relative error:{} of output {}, benchmark {} exceeding 1%".format(station, comp, im, relative_error, float2, float1)
                        errors += e + '\n'
                        failed += 1
                    else:
                        passed += 1
                except KeyError:
                        w = '{} does not exists in benchmark_dict'.format(im)
                        warning += w + '\n'
                except ValueError:
                    w = "Benchmark {} value is None".format(im)
                    warning += w + '\n'
    if warning:
        print("Warning: {}".format(warning))
    print("{} passed\n{} failed".format(passed, failed))

    return errors


def remove_folder(folder_path):
    """
    :param folder_path: eg. OUTPUTBINARY
    :return:
    """
    try:
        shutil.rmtree(folder_path)
    except (IOError, OSError):
        raise


def remove_file(file_path):
    try:
        os.remove(file_path)
    except (IOError, OSError):
        raise


def run_test_single_output_file(test_output_subfolder):
    errors = ''
    for f in os.listdir(test_output_subfolder):
        f_path = os.path.join(test_output_subfolder, f)
        errors += run_test_calculate_ims(f_path)
    return errors


def test_binary_single_output_file():
    errors = run_test_single_output_file(OUTPUT_BINARY_SUBDIR)
    assert errors == ''
    remove_folder(OUTPUT_BINARY_SUBDIR)


def test_ascii_single_output_file():
    errors = run_test_single_output_file(OUTPUT_ASCII_SUBDIR)
    assert errors == ''
    remove_folder(OUTPUT_ASCII_SUBDIR)


def test_binary_output():
    errors = run_test_calculate_ims(OUTPUT_BINARY_FILE)
    assert errors == ''
    remove_file(OUTPUT_BINARY_FILE)
    remove_file(OUTPUT_BINARY_META)


def test_ascii_output():
    errors = run_test_calculate_ims(OUTPUT_ASCII_FILE)
    assert errors == ''
    remove_file(OUTPUT_ASCII_FILE)
    remove_file(OUTPUT_ASCII_META)


# # This function should only be used when you want to re-generate then benchmark file
# # sample_bench_path = '/home/yzh231/new_im_sim_benchmark'
# # sample_bench_path is the path to the folder that contains all single_compoent_summary benchmark csvs
# def write_benchmark_csv(sample_bench_path):
#     exts = COMP_DICT.values()
#     bench_files = ['new_im_sim_{}.csv'.format(ext) for ext in exts]
#     bench_bufs = []
#     for bench_file in bench_files:
#         whole_bench_path = os.path.join(sample_bench_path, bench_file)
#         with open(whole_bench_path, 'r') as whole:
#             buf = whole.readlines()
#             bench_bufs.append(buf)
#
#     head_wirtten = False
#     with open(sample_bench_path, 'w') as file_writer:
#         for buf in bench_bufs:
#             if not head_wirtten:
#                 file_writer.write(buf[0])
#                 head_wirtten = True
#             for line in buf[1:]:
#                 line_seg = line.split(',')
#                 if line_seg[0] in STATIONS:
#                     file_writer.write(line)
#
# write_benchmark_csv(BENCHMARK)

