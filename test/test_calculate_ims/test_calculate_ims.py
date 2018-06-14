import os

TEST_FOLDER = os.path.abspath(os.path.dirname(__file__))
SCRIPT = os.path.abspath(os.path.join(TEST_FOLDER, '..', '..', 'calculate_ims.py'))
print(SCRIPT)

WHOLE_BENCH_PATH = '/home/yzh231/new_im_sim_benchmark'
INPUT_DIR = os.path.join(TEST_FOLDER,'sample1','input')
INPUT_FILES = os.path.join(INPUT_DIR, 'single_files')
BENCHMARK = '/home/yzh231/new_im_sim_benchmark/new_im_sim_benchmark.csv'
print(BENCHMARK)

OUTPUT_DIR = os.path.join(TEST_FOLDER,'sample1','output')
IDENTIFIER = 'new_im_sim'
OUTPUT_FILE = '/home/yzh231/new_im_sim_benchmark_ascii/all_station_ims/all_station_ims.csv'
print OUTPUT_FILE
OUTPUT_SUBDIR = '/home/yzh231/new_im_sim_benchmark_ascii/new_im_sim/all_station_ims/stations/'

COMP = 'ellipsis'
STATIONS = '2002199 GRY 00020d3 UNK CASH CFW DLX LSRC EWZ PEAA'
ERROR_LIMIT = 0.01

COMP_DICT = {'90': '090', 'geom': 'geom', '0': '000', 'ver': 'ver'}


def get_result_dict(sample_path):
    result_dict = {}
    comp = None
    with open(sample_path, 'r') as result_reader:
        buf = result_reader.readlines()
        #print("bubububububububub",len(buf[0].strip().split(',')))
        measures = buf[0].strip().split(',')[2:]
        for i in range(len(measures)):
            if 'pSA' in measures[i]:
                pre, suf = measures[i].split('_')
                num = '{:.9f}'.format(float(suf.replace('p', '.')))
                measures[i] = pre + '_' + num
        for line in buf[1:]:
            line_seg = line.strip().split(',')
            #print("dafsafdfad",len(line_seg))
            station = line_seg[0]
            if station == 'station_name':
                print line
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


def test_calculate_ims():
    benchmark_dict = get_result_dict(BENCHMARK)
    output_dict = get_result_dict(OUTPUT_FILE)
    passed = 0
    failed = 0
    for station in benchmark_dict.keys():
        for comp in benchmark_dict[station].keys():
            for im in benchmark_dict[station][comp].keys():
                try:
                    float1 = float(benchmark_dict[station][comp][im])
                    float2 = float(output_dict[station][COMP_DICT[comp]][im])
                    relative_error = abs(float1 - float2) / float1
                    if relative_error > ERROR_LIMIT:
                        print("{}: {}: {} relative error:{} of output {}, benchmark {} exceeding 1%".format(station, comp, im, relative_error, float2, float1))
                        failed += 1
                    else:
                        passed += 1
                except KeyError:
                        print('{} does not exists in output_dict'.format(im))
                        continue
                except ValueError:
                    print("Benchmark {} value is None".format(im))
                    continue
    print("{} passed\n{} failed".format(passed, failed))


def main():
    test_calculate_ims()


if __name__ == '__main__':
    main()


# def write_benchmark_csv(sample_bench_path):
#     exts = COMP_DICT.values()
#     bench_files = ['new_im_sim_{}.csv'.format(ext) for ext in exts]
#     bench_bufs = []
#     for bench_file in bench_files:
#         whole_bench_path = os.path.join(WHOLE_BENCH_PATH, bench_file)
#         with open(whole_bench_path, 'r') as whole:
#             buf = whole.readlines()
#             bench_bufs.append(buf)
#
#     with open(sample_bench_path, 'w') as file_writer:
#         for buf in bench_bufs:
#
#             file_writer.write(buf[0])
#
#             for line in buf[1:]:
#                 line_seg = line.split(',')
#                 if line_seg[0] in STATIONS:
#                     file_writer.write(line)

# write_benchmark_csv(BENCHMARK)
