IM Calculation - Intensity measure calculations

(Based on https://wiki.canterbury.ac.nz/download/attachments/58458136/CodeVersioning_v18p2.pdf?version=1&modificationDate=1519269238437&api=v2 )
Unreleased
Fixed

Changed

Added

[19.6.1] - 2019-06-06  
Added  
    - setup.py to make IM_calculation a package  
Changed  
    - Modified rrup calculation for significantly improved performance using numba  
    - General tidy up/refactor when converting to package  
    - Updated tests  

[18.9.1] - 2018-09-20
Added
    -
Changed
    renamed generate_sl.py to submit_imcalc.py
    modified submit_imcalc.py and checkpoint.py to be able to integrate with the slurm_gm_workflow
    moved submit_imcalc.py, checkpoint.py and im_calc_sl.template to slurm_gm_workflow
    

[18.7.2] - 2018-07-11
Added
    -
Changed
    modified generate_split_sl script to make optional args acutally optional


[18.7.1] - 2018-07-10
Added
    check_point.py
    generate_split_sl.py
Changed
    modified calculate_im script to output '_imcalc.info' metadata file instead of '.info' file
    modified im_calc_sl.template to work with the current input Cybershake sim/obs/rrup data folder structure
    modified calculate_rrups script with added option '-fd' to take a list of in_sim_domain stations instead of whole nz stations 


[18.6.4] - 2018-06-19
Added
    test_calculate_ims.py
Changed
    modified calculate_im script with added -s --simple_output option to only output summary csv
    Renamed setup.py to setup_rspectra.py


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



