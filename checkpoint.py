import os
import glob

# Examples:
# sim_waveform_dirs =
# ['/nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p6/Runs/test/Kelly/BB/Cant1D_v3-midQ_OneRay_hfnp2mm+_rvf0p8_sd50_k0p045/Kelly_HYP20-29_S1434',
# '/nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p6/Runs/test/Kelly/BB/Cant1D_v3-midQ_OneRay_hfnp2mm+_rvf0p8_sd50_k0p045/Kelly_HYP29-29_S1524',
# '/nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p6/Runs/test/Kelly/BB/Cant1D_v3-midQ_OneRay_hfnp2mm+_rvf0p8_sd50_k0p045/Kelly_HYP07-29_S1304']
# dire = '/nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p6/Runs/test/Kelly/BB/Cant1D_v3-midQ_OneRay_hfnp2mm+_rvf0p8_sd50_k0p045/Kelly_HYP20-29_S1434'
# output_sim_dir = /nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p6/Runs/test/Kelly/BB/Cant1D_v3-midQ_OneRay_hfnp2mm+_rvf0p8_sd50_k0p045/Kelly_HYP20-29_S1434/../../../IM_calc/Kelly_HYP20-29_S1434/


def checkpoint_sim_obs(waveform_dirs, relative_path_to_im_calc):
    """
    Checkpoint for simulation and observed waveform dirs
    :param waveform_dirs:a list of both completed and not completed sim/obs waveform dirs
    :param relative_path_to_im_calc: relative path from a single waveform dir to the IM_Calc output folder,
                                     should be consistent with the path defined in im_calc_sl.template.
    :return: a list of not completed waveform dirs
    """
    for directory in waveform_dirs[:]:
        dir_name = directory.split('/')[-1]
        if dir_name == 'IM_calc':
            waveform_dirs.remove(directory)
        else:
            output_dir = os.path.join(directory, relative_path_to_im_calc, dir_name)
            if os.path.isdir(output_dir):  # if output dir exists
                sum_csv = glob.glob1(output_dir, '*.csv')
                meta = glob.glob1(output_dir, '*imcalc.info')
                # if sum_csv and meta are not empty lists('.csv' and '_imcalc.info' files present)
                # then we think im calc on the corresponding dir is completed and hence remove
                if sum_csv and meta:
                    waveform_dirs.remove(directory)
    return waveform_dirs


def checkpoint_rrup(output_dir, srf_files):
    """
    Checkpoint for rrups
    :param output_dir: user input output dir to store computed rrups
    :param srf_files: a list of both completed and not completed srf files for computing rrups.
    :return: a list of not completed srf files
    """
    for srf in srf_files[:]:
        srf_name = srf.split('/')[-1].split('.')[0]
        output_path = os.path.join(output_dir, srf_name + '.csv')
        if os.path.isfile(output_path):
            srf_files.remove(srf)
    return srf_files

