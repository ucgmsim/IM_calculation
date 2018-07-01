from jinja2 import Template, Environment, FileSystemLoader
import argparse
import os
import glob
import check_point
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


def generate_sl(sim_dirs, obs_dirs, station_file, rrup_files, output_dir, i,np=8):
    path = os.path.dirname(os.path.realpath(TEMPLATE_NAME))
    j2_env = Environment(loader=FileSystemLoader(path), trim_blocks=True)

    context = j2_env.get_template(TEMPLATE_NAME).render(
        time=TIME,
        sim_dirs=sim_dirs, obs_dirs=obs_dirs,
        rrup_files=rrup_files, station_file=station_file,
        output_dir=output_dir, np=np)
    print("context")
    with open('im_calc_{}.sl'.format(i), 'w') as sl:
        print("writing",'im_calc_{}.sl'.format(i))
        sl.write(context)
        

def get_basename_without_ext(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_fault_name(run_name):
    return run_name.split('_')[0]


def split_and_generate_slurms(sim_dirs, max_lines,station_file, output_dir, processes):
    total_sim_lines = len(sim_dirs)
    print(total_sim_lines)
    i = 0
    while i < total_sim_lines:
        last_line_index = i +  max_lines
        if 0 <= last_line_index - total_sim_lines <= max_lines:
            last_line_index = total_sim_lines
            print("encountering last line in total")
        print(i,last_line_index)
        generate_sl(sim_dirs[i: last_line_index], [], station_file, [], output_dir,i, processes)
        print("generated a sl")
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
    parser.add_argument('-ml', '--max_line', default=33, help="maximun number of lines in a slurm script")
    parser.add_argument('rrup_output_dir', help="directory containing rupture distances output")

    args = parser.parse_args()

    output_dir = args.rrup_output_dir
    utils.setup_dir(output_dir)

    max_lines = args.max_line
    # /nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p5/Runs
    sim_waveform_dirs = glob.glob(os.path.join(args.sim_dir, '*/BB/*/*'))
    # print("pre sim",len(sim_waveform_dirs1))
    sim_waveform_dirs = check_point.check_point_merged(sim_waveform_dirs,
                                                       '../../../IM_calc/')  # return dirs that are not calculated yet
    #  print("after removing",len(sim_waveform_dirs))

    sim_run_names = map(os.path.basename, sim_waveform_dirs)
    sim_faults = map(get_fault_name, sim_run_names)
    sim_dirs = zip(sim_waveform_dirs, sim_run_names, sim_faults)

    srf_files = glob.glob(os.path.join(args.srf_dir, "*/Srf/*.srf"))
    #  print("srf_files are:", srf_files)
    srf_files = check_point.check_point_rrup(output_dir, srf_files)
    run_names = map(get_basename_without_ext, srf_files)
    rrup_files = zip(srf_files, run_names)

    obs_waveform_dirs = glob.glob(os.path.join(args.obs_dir, '*'))
    # print("obs_waveform_dirs", obs_waveform_dirs)
    obs_waveform_dirs = check_point.check_point_merged(obs_waveform_dirs, '../IM_calc/')
    obs_run_names = map(os.path.basename, obs_waveform_dirs)
    obs_faults = map(get_fault_name, obs_run_names)
    obs_dirs = zip(obs_waveform_dirs, obs_run_names, obs_faults)

    split_and_generate_slurms(sim_dirs, max_lines,args.station_file, output_dir, args.processes)

   # generate_sl(sim_dirs, obs_dirs, args.station_file, rrup_files, output_dir, args.processes)


if __name__ == '__main__':
    main()


