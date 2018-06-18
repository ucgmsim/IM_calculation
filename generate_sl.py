from jinja2 import Template, Environment, FileSystemLoader
import argparse
import os
import glob

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


def generate_sl(sim_dirs, obs_dirs, station_file, rrup_files, output_dir, np=8):
    path = os.path.dirname(os.path.realpath(TEMPLATE_NAME))
    j2_env = Environment(loader=FileSystemLoader(path), trim_blocks=True)

    print j2_env.get_template(TEMPLATE_NAME).render(
        time=TIME,
        sim_dirs=sim_dirs, obs_dirs=obs_dirs,
        rrup_files=rrup_files, station_file=station_file,
        output_dir=output_dir, np=np)


def get_basename_without_ext(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_fault_name(run_name):
    return run_name.split('_')[0]


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
    parser.add_argument('rrup_output_dir', help="directory containing rupture distances output")

    args = parser.parse_args()

    # /nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p5/Runs
    sim_waveform_dirs = glob.glob(os.path.join(args.sim_dir, '*/BB/*/*'))
    sim_run_names = map(os.path.basename, sim_waveform_dirs)
    sim_faults = map(get_fault_name, sim_run_names)
    sim_dirs = zip(sim_waveform_dirs, sim_run_names, sim_faults)

    srf_files = glob.glob(os.path.join(args.srf_dir, "*/Srf/*.srf"))
    run_names = map(get_basename_without_ext, srf_files)
    rrup_files = zip(srf_files, run_names)

    obs_waveform_dirs = glob.glob(os.path.join(args.obs_dir, '*'))
    obs_run_names = map(os.path.basename, obs_waveform_dirs)
    obs_faults = map(get_fault_name, obs_run_names)
    obs_dirs = zip(obs_waveform_dirs, obs_run_names, obs_faults)

    output_dir = args.rrup_output_dir

    generate_sl(sim_dirs, obs_dirs, args.station_file, rrup_files,
                output_dir, args.processes)


if __name__ == '__main__':
    main()
