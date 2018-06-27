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
        meta = glob.glob1(output_dir, '*.info')
        if meta:
            completed = True
    return completed


# sim_waveform_dirs =
# ['/nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p6/Runs/test/Kelly/BB/Cant1D_v3-midQ_OneRay_hfnp2mm+_rvf0p8_sd50_k0p045/Kelly_HYP20-29_S1434',
# '/nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p6/Runs/test/Kelly/BB/Cant1D_v3-midQ_OneRay_hfnp2mm+_rvf0p8_sd50_k0p045/Kelly_HYP29-29_S1524',
# '/nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p6/Runs/test/Kelly/BB/Cant1D_v3-midQ_OneRay_hfnp2mm+_rvf0p8_sd50_k0p045/Kelly_HYP07-29_S1304']

# dire = '/nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p6/Runs/test/Kelly/BB/Cant1D_v3-midQ_OneRay_hfnp2mm+_rvf0p8_sd50_k0p045/Kelly_HYP20-29_S1434'

# output_sim_dir = /nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p6/Runs/test/Kelly/BB/Cant1D_v3-midQ_OneRay_hfnp2mm+_rvf0p8_sd50_k0p045/Kelly_HYP20-29_S1434/../../../IM_calc/Kelly_HYP20-29_S1434/


def check_point(sim_waveform_dirs):
    removed=0
    count = 0
    j = 0
    e = 0
    for dire in sim_waveform_dirs[:]:
        dire_name = dire.split('/')[-1]
        output_dir = os.path.join(dire, '../../../IM_calc/', dire_name)
        exists = check_output_exits(output_dir)
       # print("{} exists {}".format(output_dir, exists))
        if exists:
            e += 1
            is_completed = check_completion(output_dir) 
	#   print("{} completed {}".format(output_dir, is_completed))
            if is_completed:
                sim_waveform_dirs.remove(dire)
                removed += 1
    print("exists", e, "removed", removed, "now sim wave form dir len is ", len(sim_waveform_dirs))
    return sim_waveform_dirs

#TODO merge with check_point
def check_point_obs(obs_waveform_dirs):
    removed=0
    count = 0
    j = 0
    e = 0
    for dire in obs_waveform_dirs[:]:
        dire_name = dire.split('/')[-1]
        output_dir = os.path.join(dire, '../IM_calc/', dire_name)
        print("output_dir", output_dir)
        exists = check_output_exits(output_dir)
       # print("{} exists {}".format(output_dir, exists))
        if exists:
            e += 1
            is_completed = check_completion(output_dir)
        #   print("{} completed {}".format(output_dir, is_completed))
            if is_completed:
                obs_waveform_dirs.remove(dire)
                removed += 1
    print("exists", e, "removed", removed, "now obs wave form dir len is ", len(obs_waveform_dirs))
    return obs_waveform_dirs



def get_header_size(sl_template_path):
    with open(sl_template_path, 'r') as sl:
        lines = sl.readlines()
        header_size = 0
        for line in lines:
            if not line.startswith('echo'):
                header_size += 1
            else:
                return header_size


def split_slurms(sim_waveform_dirs, sl_template_path='im_calc_sl.template', line_max=1027):
    header_size = get_header_size(sl_template_path)
    print("header size is ", header_size)
    total_lines = len(sim_waveform_dirs) + header_size
    total_scripts = total_lines // line_max
    if total_scripts * line_max < total_lines:
        total_scripts += 1
    return total_scripts


# obs folder /nesi/transit/nesi00213/RunFolder/leer/scratch
