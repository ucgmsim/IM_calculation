# IM_calculation

To calculate rrups:

```
python usage: calculate_rrups.py [-h] [-np PROCESSES] [-s STATIONS [STATIONS ...]]
                                 [-o OUTPUT]
                                 station_file srf_file
```

To calculate IMs:

We need to first setup Cython script rspectra.pyx so that we can successfully run pSA calculations inside calculate_ims.py.
In the terminal, type:

$ cd IM_calculation

$ python setup.py build_ext --inplace

```
usage: calculate_ims.py [-h] [-o OUTPUT] [-m IM [IM ...]]
                           [-p PERIOD [PERIOD ...]] [-e]
                           [-n STATION_NAMES [STATION_NAMES ...]]
                           [-c COMPONENT]
                           input_path {a,b}

positional arguments:
  input_path            path to input bb binary file eg./home/melody/BB.bin
  {a,b}                 Please type 'a'(ascii) or 'b'(binary) to indicate the
                        type of input file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        path to output folder that stores the computed
                        measures. Default to /computed_measures/
  -m IM [IM ...], --im IM [IM ...]
                        Please specify im measure(s) seperated by a space(if
                        more than one). eg: PGV PGA CAV. Available and default
                        measures are: PGV, PGA, CAV, AI, Ds575, Ds595, pSA
  -p PERIOD [PERIOD ...], --period PERIOD [PERIOD ...]
                        Please provide pSA period(s) separated by a space. eg:
                        0.02 0.05 0.1. Available and default periods are:0.02
                        0.05 0.1 0.2 0.3 0.4 0.5 0.75 1.0 2.0 3.0 4.0 5.0 7.5
                        10.0
  -e, --extended_period
                        Please add '-e' to indicate the use of extended(100)
                        pSA periods. Default not using
  -n STATION_NAMES [STATION_NAMES ...], --station_names STATION_NAMES [STATION_NAMES ...]
                        Please provide a station name(s) seperated by a space.
                        eg: 112A 113A
  -c COMPONENT, --component COMPONENT
                        Please provide the velocity/acc component(s) you want
                        to calculate eperated by a spave. eg.000 090 ver
  -np PROCESS, --process PROCESS
                        Please provide the number of processers

```
