IM Calculation - Intensity measure calculations

(Based on https://wiki.canterbury.ac.nz/download/attachments/58458136/CodeVersioning_v18p2.pdf?version=1&modificationDate=1519269238437&api=v2 )
Unreleased
Fixed

Changed

Added


[18.6.4] - 2018-06-19
Added
    test_calculate_ims.py
Changed
    modified calculate_im script with added -s --simple_output option to only output summary csv


[18.6.3] - 2018-06-14
Added
    Slurm script generation to aide submission for the job to Kupe/Maui/Mahuika for binary and ascii workflows
    aggregation of IMs across realisations script
    Option to select acc units for each run
Changed
    modified calculate_im script to use unique ascending periods when concatenating with extended periods
    modified calculate_im script to use sorted stations when writing result csvs


[18.6.2] - 2018-06-11
Added
    -
Changed
    modified calculate_im script to be able to generate meta_data file
    modified README to display the new help message

[18.6.1] - 2018-06-07
Added
    calculate_im script
    setup.py
Changed
    modified pool_wrapper to be able to run with 1 processor
    modified intensity_measures to be able to process nd array
    modified read_waveform to be able to read binary file


[18.5.1] - 2017-05-24 -- Initial Version
Added
    rrup calculation script
    R_x template code
Changed
    modified rrup.csv output to conform to the template on wiki



