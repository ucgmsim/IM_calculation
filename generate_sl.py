from jinja2 import Template, Environment, FileSystemLoader
import argparse
import os
import glob
import checkpoint
from qcore import utils

TEMPLATE_NAME = 'im_calc_sl.template'
TIME = '00:30:00'
DEFAULT_N_PROCESSES = 40


# TODO: calculate wall-clock time
# TODO: read fd*.ll file to limit the stations that rrups is calculated for
# TODO: option for binary workflow
# TODO: handle optional arguments correctly
# TODO: rrup output_dir the csv to each individual simulation folder
# TODO: one rupture distance calc per fault
# TODO: remove relative paths on sl.template


def generate_sl(sim_dirs, obs_dirs, station_file, rrup_files, output_dir, prefix, i, np=8):
    path = os.path.dirname(os.path.realpath(TEMPLATE_NAME))
    j2_env = Environment(loader=FileSystemLoader(path), trim_blocks=True)

    context = j2_env.get_template(TEMPLATE_NAME).render(
        time=TIME,
        sim_dirs=sim_dirs, obs_dirs=obs_dirs,
        rrup_files=rrup_files, station_file=station_file,
        output_dir=output_dir, np=np)
    sl_name = '{}_im_calc_{}.sl'.format(prefix, i)
    with open(sl_name, 'w') as sl:
        print("writing {}".format(sl_name))
        sl.write(context)


def get_basename_without_ext(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_fault_name(run_name):
    return run_name.split('_')[0]


def split_and_generate_slurms(sim_dirs, obs_dirs, station_file, rrup_files, output_dir, processes, max_lines, prefix):
    total_dir_lines = 0
    if sim_dirs != []:
        total_dir_lines = len(sim_dirs)
    elif obs_dirs != []:
        total_dir_lines = len(obs_dirs)
    elif rrup_files != []:
        total_dir_lines = len(rrup_files)
    i = 0
    while i < total_dir_lines:
        last_line_index = i + max_lines
        if 0 <= last_line_index - total_dir_lines <= max_lines:
            last_line_index = total_dir_lines
        generate_sl(sim_dirs[i: last_line_index], obs_dirs[i: last_line_index], station_file,
                    rrup_files[i: last_line_index], output_dir, prefix, i, processes)
        i += max_lines


def main():
    parser = argparse.ArgumentParser(description="Prints out a slurm script to run IM Calculation over a run-group")
    parser.add_argument('-s', '--sim_dir',
                        help="Path to sim-run-group containing faults and acceleration in the subfolder */BB/*/*")
    parser.add_argument('-o', '--obs_dir',
                        help="Path to obs-run-group containing faults and accelerations in the subfolder */*/accBB")
    parser.add_argument('-srf', '--srf_dir',
                        help="Path to run-group containing the srf files in the path matching */Srf/*.srf")
    parser.add_argument('-ll', '--station_file',
                        help="Path to a single station file for ruputure distance calculations")
    parser.add_argument('-np', '--processes', default=DEFAULT_N_PROCESSES, help="number of processors to use")
    parser.add_argument('-ml', '--max_line', default=33, type=int, help="maximum number of lines in a slurm script")
    parser.add_argument('rrup_output_dir', help="directory containing rupture distances output")

    args = parser.parse_args()

    output_dir = args.rrup_output_dir
    utils.setup_dir(output_dir)

    max_lines = args.max_line
    if max_lines <= 0:
        parser.error("-ml argument should come with a number that is 0 < -ml <= (max_lines-header_and_other_prints) allowed by slurm")

    station_file = args.station_file
    processes = args.processes

    # sim_dir = /nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p5/Runs
    if args.sim_dir is not None:
        sim_waveform_dirs = glob.glob(os.path.join(args.sim_dir, '*/BB/*/*'))
        sim_waveform_dirs = checkpoint.checkpoint_sim_obs(sim_waveform_dirs, '../../../IM_calc/')  # return dirs that are not calculated yet
        sim_run_names = map(os.path.basename, sim_waveform_dirs)
        sim_faults = map(get_fault_name, sim_run_names)
        sim_dirs = zip(sim_waveform_dirs, sim_run_names, sim_faults)
        # sim
        split_and_generate_slurms(sim_dirs, [], station_file, [], output_dir, processes, max_lines, 'sim')

    if args.srf_dir is not None:
        srf_files = glob.glob(os.path.join(args.srf_dir, "*/Srf/*.srf"))
        srf_files = checkpoint.checkpoint_rrup(output_dir, srf_files)
        run_names = map(get_basename_without_ext, srf_files)
        rrup_files = zip(srf_files, run_names)
        # rrup
        split_and_generate_slurms([], [], station_file, rrup_files, output_dir, processes, max_lines, 'rrup')

    if args.obs_dir is not None:
        obs_waveform_dirs = glob.glob(os.path.join(args.obs_dir, '*'))
        obs_waveform_dirs = checkpoint.checkpoint_sim_obs(obs_waveform_dirs, '../IM_calc/')
        obs_run_names = map(os.path.basename, obs_waveform_dirs)
        obs_faults = map(get_fault_name, obs_run_names)
        obs_dirs = zip(obs_waveform_dirs, obs_run_names, obs_faults)
        # obs
        split_and_generate_slurms([], obs_dirs, station_file, [], output_dir, processes, max_lines, 'obs')


if __name__ == '__main__':
    main()



