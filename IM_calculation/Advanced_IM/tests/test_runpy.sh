#!/bin/bash
#out_dir=/nesi/nobackup/nesi00213/RunFolder/ykh22/test_adv_im/v19p5p8_no_gr_SAC_3and9/Runs/3792018/3792018/test_station/tmp/test_ATC12
if [[ $# -lt 1 ]];then
    echo "please provide 1. the model name 2. level of magnitude(small, moderate,large) 3. path for output (optional)"
    exit 1
fi

#get the absolute path of this script
DIR=$( dirname "$( realpath "${BASH_SOURCE[0]}")" )

model_name=${1?Error : argument one cannot be Null}
#make sure the model name provided has a run.py
runpy_path=$gmsim/IM_calculation/IM_calculation/Advanced_IM/Models/run_2D.py
#runpy_path=$gmsim/IM_calculation/IM_calculation/Advanced_IM/Models/run_2D.py

# check if 


if [[ ! -f $runpy_path ]];then
    echo "cannot find $runpy_path"
    echo "please make sure the name of model provided are correct"
    exit 1
fi

mag_catagory=${2:-small}
waveform_dir=$DIR/waveforms/$mag_catagory/
station_name=`basename $waveform_dir/*.000 | cut -d. -f1`
if [[ $? != 0 ]];then
    exit
fi

out_dir=${3:-$DIR/tmp/$model_name}
if [[ -f $out_dir ]] || [[ -d $out_dir ]];then
    echo "$out_dir exist, please remove or backup the previous runs"
    exit 2
else
    mkdir -p $out_dir
fi

runtime_fmt="%Y-%m-%d_%H:%M:%S"

start_time=`date +$runtime_fmt`
python $runpy_path $model_name $waveform_dir/$station_name.000 $waveform_dir/$station_name.090 $waveform_dir/$station_name.ver $out_dir --OpenSees_path /nesi/project/nesi00213/opt/maui/tmp/OpenSees 2>&1 1>/dev/null

end_time=`date +$runtime_fmt`
#echo start_time $start_time >> time_$model_name
#echo end_time $end_time >> time_$model_name

