##############################################################################################################
# Reagan Chandramohan                                                                                        #
# John A. Blume Earthquake Engineering Center                                                                #
# Stanford University                                                                                        #
# Last edited: 28-Aug-2015                                                                                   #
##############################################################################################################

# Run an incremental dynamic analysis on the specified model number using multiple cores.
# Requires OpenSeesMP (parallel version). Will not run on OpenSees (sequential version).
# For best results, run with 10 times as many processors as there are ground motions to be run + 1.

##############################################################################################################

# Initialize the total number of processors and the id of this processor
set numprocs [getNP]
set this_procid [getPID]

# If only one processor is available, display a message and halt execution
if {$numprocs == 1} {
    puts "No free processors available. Job must be run with at least 2 processors."
    return
}

# Check the command line arguments
if {$argc != 1} {
    puts "Requires one argument: Model number to run"
    return
}

# Compute the model directory from the model number received as command line argument
set modelpath Models
set modeldirs [lsort [glob -directory $modelpath -type d *]]
set modeldir [lindex $modeldirs [expr {$argv - 1}]]
set slash_loc [string first / $modeldir]
set modelname [string range $modeldir [expr {$slash_loc + 1}] end]

# Initialize the list of ground motion folders to be run
set inpath ../../Ground_Motions
set indirlist FEMA_P695_Far_Field_Long

# Define the output path
set outpath Analysis_Results/$modelname/IDA

# Define the collapse peak story drift ratio
set col_drift 0.10

# Tasks to be executed by the master processor (processor number 0)
if {$this_procid == 0} {

    # Read the fundamental mode period of the structure in seconds (used to read the appropriate SaT1 file
    # for scaling)
    set periods_file [open $modeldir/Model_Info/periods.out r]
    gets $periods_file T1
    close $periods_file

    # Create the list of ground motions to be run by looping over all the ground motion folders
    set serial 0
    foreach indir $indirlist {

        # Import information about each ground motion from the "GMInfo.txt" file and their Sa(T1,5%) values
        # from the SaT1 file corresponding to the fundamental period of the structure, and add them to the
        # "gminfo_dict" dictionary
        set gminfofile [open $inpath/$indir/GMInfo.txt r]
        set sat1file [open $inpath/$indir/SaT1/SaT1_5_[format "%.3f" $T1]s.out r]

        while {[gets $gminfofile line1] >= 0 && [gets $sat1file line2] >= 0} {

            # Read the filename and dt
            set filename [lindex $line1 1]
            set dt [lindex $line1 2]

            # Count the number of points
            set numpts 0
            set gmfile [open $inpath/$indir/$filename r]
            while {[gets $gmfile line1] >= 0} {
                incr numpts
            }
            close $gmfile

            # Add the ground motion information to "gminfo_dict"
            dict set gminfo_dict $serial indir $indir
            dict set gminfo_dict $serial filename $filename
            dict set gminfo_dict $serial dt $dt
            dict set gminfo_dict $serial numpts $numpts

            # Add the Sa(T1,5%) value of the ground motion to "gminfo_dict"
            dict set gminfo_dict $serial sat1_gm $line2

            # Add the output filename to "gminfo_dict"
            set filename_length [string length $filename]
            dict set gminfo_dict $serial outfilename [string range $filename 0 [expr {$filename_length - 4}]]

            incr serial
        }

        close $gminfofile
        close $sat1file
    }

    # Set the total number of ground motions to be run
    set numgms $serial

    # Define the number of ground motion scales to be forked and run simultaneously and the spacing of the
    # ground motion scales in the coarse and fine stages
    set delta_sat1_coarse 0.10
    set numforks_coarse 10
    set delta_sat1_fine 0.01
    set numforks_fine [expr {round($delta_sat1_coarse/$delta_sat1_fine) - 1}]

    # Function to display the ground motion number that has caused collapse and the time when it completed
    # execution
    proc display_status gm {
        set time [clock seconds]
        puts "Ground Motion #[expr {$gm + 1}] collapsed - [clock format $time -format {%D %H:%M:%S}]"
    }

    # Function to send a message to all slave processors to terminate execution
    proc terminate_slave_procs numprocs {
        for {set procid 1} {$procid < $numprocs} {incr procid} {
            send -pid $procid "DONE"
        }
    }

    # Loop over all ground motions and initialize the data structures that are used to control the analysis
    for {set gm 0} {$gm < $numgms} {incr gm} {
        
        # Check if previous, incomplete analysis results exist for this ground motion by checking whether the
        # "completed_scales.txt" file exists in the output folder corresponding to the current ground motion.
        set gminfo [dict get $gminfo_dict $gm]
        dict with gminfo {

            # If previous analysis results exist, initialize the data structures based on the contents of the
            # "completed_scales.txt" file
            if {[file exists $outpath/$indir/$outfilename/completed_scales.txt]} {

                # "ida_curves" is a dictionary that stores the SaT1 values and corresponding computed peak
                # story drifts for all ground motions. This is finally used to write the "ida_curve.txt" file
                # for each ground motion.
                dict set ida_curves $gm 0.0 0.0

                # "lowest_fork_collapse" is an array that stores the lowest scale that caused the structure to
                # collapse for each ground motion
                set lowest_fork_collapse($gm) inf

                # Temporary variable to store the lowest scale that caused the structure to collpase in the
                # coarse stage
                set lowest_fork_collapse_coarse inf

                # Counter for the number of scales run at each tier. Tier 1 corresponds to the coarse stage,
                # while tier 2 corresponds to the fine stage.
                dict set numscales 1 0
                dict set numscales 2 0

                # Loop through the contents of the "completed_scales.txt" file
                set completed_scales_file [open $outpath/$indir/$outfilename/completed_scales.txt r]
                while {[gets $completed_scales_file line] >= 0} {
                    
                    # Parse the contents of each line
                    set sat1 [lindex $line 0]
                    set drift [lindex $line 1]
                    set tier [lindex $line 2]

                    # Increment the counter corresponding to the tier, and add the scale to the list of
                    # completed scales
                    dict incr numscales $tier
                    lappend completed_scales($gm) [format "%.3f" $sat1]

                    # Based on whether the drift is lesser or greater than the collapse drift, either add the
                    # line to the "ida_curves" dictionary or update the "lowest_forks_collapse" entry
                    # corresponding to the current ground motion
                    if {$drift < $col_drift} {
                        dict set ida_curves $gm $sat1 $drift
                    } else {
                        set lowest_fork_collapse($gm) [expr {min($lowest_fork_collapse($gm), $sat1)}]

                        # If the scale is in the coarse tier, update "lowest_fork_collapse_coarse"
                        if {$tier == 1} {
                            set lowest_fork_collapse_coarse $lowest_fork_collapse($gm)
                        }
                    }
                }
                close $completed_scales_file

                # If the fine stage has completed
                if {[dict get $numscales 2] == $numforks_fine} {

                    # "collapsed" is a list of length "numgms" that stores a one when a ground motion has been
                    # scaled to cause collapse and zero if it has not
                    lappend collapsed 1

                    # Display a message that the ground motion has been scaled to collapse
                    display_status $gm

                # If the coarse stage has completed, but the fine stage has not
                } elseif {[dict get $numscales 2] || (![expr {[dict get $numscales 1] % $numforks_coarse}] \
                        && $lowest_fork_collapse($gm) != inf)} {
                    lappend collapsed 0

                    # "forks_run" is an array that stores the number of simultaneous forked scales that have
                    # been run for each ground motion in the coarse or fine stages
                    set forks_run($gm) [dict get $numscales 2]

                    # "runlist" is a stack that stores all the ground motion numbers and corresponding scale
                    # factors that need to be run at a given point of time. The last element of each entry is
                    # a "tier" that can be 1 or 2 depending on whether the coarse or fine stage is being
                    # executed.
                    for {set i 1} {$i <= $numforks_fine} {incr i} {
                        set cur_scale [expr {$lowest_fork_collapse_coarse - $i*$delta_sat1_fine}]
                        
                        # Push the current scale to "runlist" only if it has not been previously run
                        if {[lsearch $completed_scales($gm) [format "%.3f" $cur_scale]] == -1} {
                            lappend runlist "$gm $cur_scale 2"
                        }
                    }

                # If the coarse stage has not completed
                } else {
                    lappend collapsed 0
                    set forks_run($gm) [dict get $numscales 1]

                    set start_fork_num [expr {($forks_run($gm)/$numforks_coarse)*$numforks_coarse}]
                    for {set i 1} {$i <= $numforks_coarse} {incr i} {
                        set cur_scale [expr {($start_fork_num + $i)*$delta_sat1_coarse}]
                        
                        # Push the current scale to "runlist" only if it has not been previously run
                        if {[lsearch $completed_scales($gm) [format "%.3f" $cur_scale]] == -1} {
                            lappend runlist "$gm $cur_scale 1"
                        }
                    }
                }

            # If previous analysis results do not exist, initialize the data structures corresponding to a new
            # analysis
            } else {
                dict set ida_curves $gm 0.0 0.0
                lappend collapsed 0
                set forks_run($gm) 0
                set lowest_fork_collapse($gm) inf

                # Push jobs corresponding to the coarse stage to "runlist"
                for {set i 1} {$i <= $numforks_coarse} {incr i} {
                    lappend runlist "$gm [expr {$i*$delta_sat1_coarse}] 1"
                }
            }
        }
    }

    # "sum_collapsed" is the sum of the "collapsed" list and stores the number of ground motions that have
    # been scaled to cause collapse
    set sum_collapsed [expr [join $collapsed +]]

    # "free_procs" is a stack containing the numbers of all the free processors at any given point of time.
    # It is initialized to contain the ids of all the slave processors.
    for {set procid 1} {$procid < $numprocs} {incr procid} {
        lappend free_procs $procid
    }

    # If the "runlist" stack is empty, display a message that all ground motions have been scaled to collapse,
    # write the "ida_curve.txt" files and terminate execution. Else, loop until all ground motions have been
    # scaled to cause collapse and the "runlist" stack is empty.
    if {![info exists runlist]} {
        puts "All ground motions have been scaled to collapse."
        terminate_slave_procs $numprocs
    } else {
        while {$sum_collapsed < $numgms || [llength $runlist]} {
            
            # Assign jobs to all free processors by popping ground motions and scales from the "runlist" stack
            # and free processors from the "free_procs" stack
            while {[llength $free_procs] && [llength $runlist]} {
                
                # Pop a ground motion and scale from the "runlist" stack
                set gm [lindex [lindex $runlist end] 0]
                set sat1 [lindex [lindex $runlist end] 1]
                set tier [lindex [lindex $runlist end] 2]
                set runlist [lreplace $runlist end end]

                # Pop a free processor from the "free_procs" stack
                set slave_procid [lindex $free_procs end]
                set free_procs [lreplace $free_procs end end]

                # Assign the job to the free processor
                send -pid $slave_procid "$gm $sat1 $tier [dict get $gminfo_dict $gm]"
            }

            # Receive the results of a completed job and push the id of the processor that finished its job to
            # the "free_procs" stack
            recv -pid ANY result
            set slave_procid [lindex $result 0]
            lappend free_procs $slave_procid

            # Parse the job results
            set gm [lindex $result 1]
            set sat1 [lindex $result 2]
            set tier [lindex $result 3]
            set drift [lindex $result 4]

            # Modify the data structures depending on whether the job was from the coarse or fine stage and
            # the results of the job
            switch $tier {
                1 {
                    # Increment the "forks_run" counter for the ground motion
                    incr forks_run($gm)

                    # If the analysis did not cause collapse, add the entry to the "ida_curves" dictionary,
                    # else update the "lowest_fork_collapse" entry for the ground motion if necessary
                    if {$drift < $col_drift} {
                        dict set ida_curves $gm $sat1 $drift
                    } else {
                        set lowest_fork_collapse($gm) [expr {min($lowest_fork_collapse($gm), $sat1)}]
                    }

                    # Write the result to the "completed_scales.txt" file in the output folder corresponding
                    # to the current ground motion
                    set gminfo [dict get $gminfo_dict $gm]
                    dict with gminfo {
                        set completed_scales_file [open $outpath/$indir/$outfilename/completed_scales.txt a]
                        puts $completed_scales_file "[format "%.3f" $sat1]\t[format "%.5f" $drift]\t$tier"
                        close $completed_scales_file
                    }

                    # If all coarse scale forks have finished running and none of the scales have caused
                    # collapse, start a new coarse stage with higher scales, else, reset the "forks_run"
                    # counter and push the ground motion with scales corresponding to the fine stage into the
                    # "runlist" stack. These fine scales are computed based on the "lowest_fork_collapse"
                    # entry for the ground motion.
                    if {[expr {$forks_run($gm) % $numforks_coarse}] == 0} {
                        if {$lowest_fork_collapse($gm) == inf} {
                            for {set i 1} {$i <= $numforks_coarse} {incr i} {
                                lappend runlist "$gm [expr {($forks_run($gm) + $i)*$delta_sat1_coarse}] 1"
                            }
                        } else {
                            set forks_run($gm) 0
                            for {set i 1} {$i <= $numforks_fine} {incr i} {
                                lappend runlist \
                                        "$gm [expr {$lowest_fork_collapse($gm) - $i*$delta_sat1_fine}] 2"
                            }
                        }
                    }
                }

                2 {
                    # Increment the "forks_run" counter for the ground motion
                    incr forks_run($gm)

                    # If the analysis did not cause collapse, add the entry to the "ida_curves" dictionary,
                    # else update the "lowest_fork_collapse" entry for the ground motion if necessary
                    if {$drift < $col_drift} {
                        dict set ida_curves $gm $sat1 $drift
                    } else {
                        set lowest_fork_collapse($gm) [expr {min($lowest_fork_collapse($gm), $sat1)}]
                    }

                    # Write the result to the "completed_scales.txt" file in the output folder corresponding
                    # to the current ground motion
                    set gminfo [dict get $gminfo_dict $gm]
                    dict with gminfo {
                        set completed_scales_file [open $outpath/$indir/$outfilename/completed_scales.txt a]
                        puts $completed_scales_file "[format "%.3f" $sat1]\t[format "%.5f" $drift]\t$tier"
                        close $completed_scales_file
                    }

                    # If all fine scale forks have finished running, update the flag corresponding to the
                    # ground motion in the "collapsed" list to indicate that the ground motion has been scaled
                    # to cause collapse and display a message
                    if {$forks_run($gm) == $numforks_fine} {
                        lset collapsed $gm 1
                        display_status $gm
                    }
                }
            }

            # Compute the total number of ground motions that have been scaled to cause collapse
            set sum_collapsed [expr [join $collapsed +]]
        }

        # Send a message to all slave processors to terminate execution
        terminate_slave_procs $numprocs
    }

    # Write the "ida_curve.txt" file for all ground motions
    for {set gm 0} {$gm < $numgms} {incr gm} {
        set gminfo [dict get $gminfo_dict $gm]
        dict with gminfo {
            
            # Open the "ida_curve.txt" file in the output folder corresponding to the current ground motion
            set ida_curve_file [open $outpath/$indir/$outfilename/ida_curve.txt w]

            # Loop through the scales run for the ground motion in sorted order
            foreach sat1 [lsort -real [dict keys [dict get $ida_curves $gm]]] {
                set drift [dict get $ida_curves $gm $sat1]
                
                # Write a scale and corresponding drift to the "ida_curve.txt" file if the scale is lesser
                # than the "lowest_fork_collapse" for the ground motion
                if {$sat1 < $lowest_fork_collapse($gm)} {
                    puts $ida_curve_file "[format "%.3f" $sat1]\t[format "%.5f" $drift]"
                }
            }
            close $ida_curve_file
        }
    }

# Tasks to be executed by all slave processors
} else {

    # Source the required files
    source max_drift_check.tcl

    # Receive a message from the master node and check if the message asks the processor to terminate
    # execution without running any jobs. This could happen if too many processors are requested. If the
    # message corresponds to a valid job, run the job, report back to the master node and wait for a new
    # message.
    recv -pid 0 command
    set status [lindex $command 0]

    for {set i 0} {$status != "DONE"} {incr i} {

        # Parse the message from the master node
        set gm [lindex $command 0]
        set sat1 [lindex $command 1]
        set tier [lindex $command 2]
        set gminfo [lrange $command 3 end]

        # Wipe previous model definitions, create the model, and run the analysis
        wipe
        source create_model.tcl
        dict with gminfo {
            source recorders_analysis_ida_mp_hpc.tcl
        }

        # Send the master node the results of the analysis and receive a new message
        send -pid 0 "$this_procid $gm $sat1 $tier $max_drift"
        recv -pid 0 command
        set status [lindex $command 0]
    }
}
