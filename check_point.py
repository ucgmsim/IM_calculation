import os
import glob


def check_output_exits(output_dir):
    if os.path.isdir(output_dir):
        exists = True
    else:
        exists = False
    return exists


def check_completion(output_dir):
    completed = False
    sum_csv = glob.glob1(output_dir, '*.csv')
    if sum_csv:
        meta = glob.glob1(output_dir, '*imcalc.info')
        if meta:
            completed = True
    return completed


# Examples:
# sim_waveform_dirs =
# ['/nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p6/Runs/test/Kelly/BB/Cant1D_v3-midQ_OneRay_hfnp2mm+_rvf0p8_sd50_k0p045/Kelly_HYP20-29_S1434',
# '/nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p6/Runs/test/Kelly/BB/Cant1D_v3-midQ_OneRay_hfnp2mm+_rvf0p8_sd50_k0p045/Kelly_HYP29-29_S1524',
# '/nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p6/Runs/test/Kelly/BB/Cant1D_v3-midQ_OneRay_hfnp2mm+_rvf0p8_sd50_k0p045/Kelly_HYP07-29_S1304']
# dire = '/nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p6/Runs/test/Kelly/BB/Cant1D_v3-midQ_OneRay_hfnp2mm+_rvf0p8_sd50_k0p045/Kelly_HYP20-29_S1434'
# output_sim_dir = /nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p6/Runs/test/Kelly/BB/Cant1D_v3-midQ_OneRay_hfnp2mm+_rvf0p8_sd50_k0p045/Kelly_HYP20-29_S1434/../../../IM_calc/Kelly_HYP20-29_S1434/


def check_point_sim_obs(waveform_dirs, relative_path_to_im_calc):
    removed = 0
    e = 0
    for dire in waveform_dirs[:]:
        dire_name = dire.split('/')[-1]
        if dire_name == 'IM_calc':
            waveform_dirs.remove(dire)
        else:
            output_dir = os.path.join(dire, relative_path_to_im_calc, dire_name)
            exists = check_output_exits(output_dir)
            if exists:
                e += 1
                is_completed = check_completion(output_dir)
                if is_completed:
                    waveform_dirs.remove(dire)
                    removed += 1
    return waveform_dirs


def check_point_rrup(output_dir, srf_files):
    removed=0
    for srf in srf_files[:]:
        srf_name = srf.split('/')[-1].split('.')[0]
        output_path = os.path.join(output_dir, srf_name + '.csv')
        if os.path.isfile(output_path):
            srf_files.remove(srf)
            removed += 1
    return srf_files

