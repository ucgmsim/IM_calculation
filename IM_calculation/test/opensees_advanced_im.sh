#!/bin/bash
if [[ $# -lt 1 ]];then
    echo "please provide 1. the model name. 2(optional): level of magnitude(small(default), moderate,large) 3(optional): path for output "
    exit 1
fi

#get the absolute path of this script
DIR=$( dirname "$( realpath "${BASH_SOURCE[0]}")" )

model_name=${1?Error : argument one cannot be Null}
#make sure the model name provided has a run.py
runpy_path=`python -c "import os;from IM_calculation.Advanced_IM import advanced_IM_factory as advf;config = advf.get_config();im_config = config[\"$model_name\"]; runpy_path = os.path.join(advf.advanced_im_dir, im_config[\"script_location\"]); print(runpy_path)"`
# check if runpy exist 
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
python $runpy_path $waveform_dir/$station_name.000 $waveform_dir/$station_name.090 $waveform_dir/$station_name.ver $out_dir --OpenSees_path /nesi/project/nesi00213/opt/maui/tmp/OpenSees 2>&1 1>/dev/null

end_time=`date +$runtime_fmt`
#echo start_time $start_time >> time_$model_name
#echo end_time $end_time >> time_$model_name

