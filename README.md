# IM_calculation
[![Build Status](https://quakecoresoft.canterbury.ac.nz/jenkins/job/IM_calculation/badge/icon?build=last:${params.ghprbActualCommit=master)](https://quakecoresoft.canterbury.ac.nz/jenkins/job/IM_calculation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

To calculate rrups:

```
python usage: calculate_rrups.py [-h] [-np PROCESSES] [-s STATIONS [STATIONS ...]]
                                 [-o OUTPUT]
                                 station_file srf_file
```

To calculate IMs:

First we need to setup Cython script rspectra.pyx so that we can successfully run pSA calculations inside calculate_ims.py.
In the terminal, type:

$ cd IM_calculation

$ python setup.py build_ext --inplace

```
usage: calculate_ims.py [-h] [-o OUTPUT_PATH] [-i IDENTIFIER] [-r RUPTURE]
                        [-t {s,o,u}] [-v VERSION] [-m IM [IM ...]]
                        [-p PERIOD [PERIOD ...]] [-e]
                        [-n STATION_NAMES [STATION_NAMES ...]] [-c COMPONENT]
                        [-np PROCESS] [-s] [-u {cm/s^2,g}]
                        input_path {a,b}

positional arguments:
  input_path            path to input bb binary file eg./home/melody/BB.bin
  {a,b}                 Please type 'a'(ascii) or 'b'(binary) to indicate the
                        type of input file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        path to output folder that stores the computed
                        measures.Folder name must not be
                        inclusive.eg.home/tt/. Default to /home/$user/
  -i IDENTIFIER, --identifier IDENTIFIER
                        Please specify the unique runname of the simulation.
                        eg.Albury_HYP01-01_S1244
  -r RUPTURE, --rupture RUPTURE
                        Please specify the rupture name of the simulation.
                        eg.Albury
  -t {s,o,u}, --run_type {s,o,u}
                        Please specify the type of the simrun. Type
                        's'(simulated) or 'o'(observed) or 'u'(unknown)
  -v VERSION, --version VERSION
                        Please specify the version of the simulation. eg.18p4
  -m IM [IM ...], --im IM [IM ...]
                        Please specify im measure(s) separated by a space(if
                        more than one). eg: PGV PGA CAV. Available and default
                        IMs are: PGA,PGV,CAV,AI,Ds575,Ds595,MMI,pSA
  -p PERIOD [PERIOD ...], --period PERIOD [PERIOD ...]
                        Please provide pSA period(s) separated by a space. eg:
                        0.02 0.05 0.1. Available and default periods are: 0.02
                        ,0.05,0.1,0.2,0.3,0.4,0.5,0.75,1.0,2.0,3.0,4.0,5.0,7.5
                        ,10.0
  -e, --extended_period
                        Please add '-e' to indicate the use of extended(100)
                        pSA periods. Default not using
  -n STATION_NAMES [STATION_NAMES ...], --station_names STATION_NAMES [STATION_NAMES ...]
                        Please provide a station name(s) separated by a space.
                        eg: 112A 113A
  -c COMPONENT, --component COMPONENT
                        Please provide the velocity/acc component(s) you want
                        to calculate eg.geom. Available compoents are:
                        090,000,ver,geom,ellipsis. ellipsis contains all 4
                        components. Default is ellipsis
  -np PROCESS, --process PROCESS
                        Please provide the number of processors. Default is 2
  --real_stats_only     Please add '--real_stats_only' to consider real stations only                        
  -s, --simple_output   Please add '-s' to indicate if you want to output the
                        big summary csv only(no single station csvs). Default
                        outputting both single station and the big summary
                        csvs
  -u {cm/s^2,g}, --units {cm/s^2,g}
                        The units that input acceleration files are in
```

To create submission script for slurm workflow, refer to the script under gm_sim_workflow repository:
https://github.com/ucgmsim/slurm_gm_workflow/blob/master/scripts/submit_imcalc.py

e.g.

```
 python submit_imcalc.py -obs ~/test_obs/IMCalcExample/ -sim runs/Runs -srf /nesi/nobackup/nesi00213/RunFolder/Cybershake/v18p6_batched/v18p6_exclude_1k_batch_6/Data/Sources -ll /scale_akl_nobackup/filesets/transit/nesi00213/StationInfo/non_uniform_whole_nz_with_real_stations-hh400_v18p6.ll -o ~/rrup_out -ml 1000 -e -s -i OtaraWest02_HYP01-21_S1244 Pahiatua_HYP01-26_S1244 -t 24:00:00
```
