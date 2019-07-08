##############################################################################################################
# Reagan Chandramohan                                                                                        #
# John A. Blume Earthquake Engineering Center                                                                #
# Stanford University                                                                                        #
# Last edited: 28-Jul-2016                                                                                   #
##############################################################################################################

# Run an incremental dynamic analysis on the specified model number in parallel.
# Requires OpenSeesMP (parallel version). Will not run on OpenSees (sequential version).

##############################################################################################################

# Initialize the total number of processes and the id of this process
set numprocs [getNP]
set this_procid [getPID]

# If only one process is available, display a message and halt execution
if {$numprocs == 1} {
    puts "No free processes available. Job must be run with at least 2 processes."
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
set indirlist FEMA_P695_Far_Field

# Define the output path
set outpath Analysis_Results/$modelname/IDA

# Define the collapse peak story drift ratio
set col_drift 0.10

# Tasks to be executed by the master process (process number 0)
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

    # Define the ground motion intensity spacing in the hunt stage and "num_bracket_stages", which defines
    # the number of analyses to be conducted in the bracket stage as 2^(num_bracket_stages) - 1
    set delta_sat1 0.02
    set num_bracket_stages 3

    # Procedure to display the ground motion number that has caused collapse and the time when it completed
    # execution
    proc display_completed gm {
        set time [clock seconds]
        puts "Ground Motion #[expr {$gm + 1}] collapsed - [clock format $time -format {%D %H:%M:%S}]"
    }

    # Procedure to send a message to all slave processes to terminate execution
    proc terminate_slave_procs numprocs {
        for {set procid 1} {$procid < $numprocs} {incr procid} {
            send -pid $procid "DONE"
        }
    }

    # Loop over all ground motions and initialize the data structures that are used to control the analysis
    for {set gm 0} {$gm < $numgms} {incr gm} {

        # "ida_curves" is a dictionary that stores the SaT1 values and corresponding computed peak story
        # drifts for all ground motions. This is finally used to write the "ida_curve.txt" file for each
        # ground motion.
        dict set ida_curves $gm 0.0 0.0

        # "current_stage" is an associative array that stores the present analysis stage of each ground motion
        # The following codes are used to identify the various possible analysis stages:
        #   0: Hunting stage
        #  >1: Bracketing stage
        #  -1: Completed
        set current_stage($gm) 0

        # "lowest_collapsed_sat1" and "highest_uncollapsed_sat1" are associative arrays that store the
        # bounding intensity values used to bracket the collapse intensity of each ground motion
        set lowest_collapsed_sat1($gm) inf
        set highest_uncollapsed_sat1($gm) 0
    }

    # Enqueue hunt stage jobs corresponding to all ground motions to the "runlist" queue
    for {set i 1} {$i <= 100} {incr i} {
        for {set gm 0} {$gm < $numgms} {incr gm} {
            lappend runlist "$gm [expr {$i*$delta_sat1}] 0"
        }
    }

    # Implement a checkpoint-restart feature that allows a previous aborted analysis to be resumed.
    # The presence of an "analysis_log.txt" file in the output folder corresponding to a ground motion
    # indicates that a previous attempt to run that ground motion was aborted.
    for {set gm 0} {$gm < $numgms} {incr gm} {
        set gminfo [dict get $gminfo_dict $gm]
        dict with gminfo {
            
            # If an "analyis_log.txt" file exists, update the data structures based on the contents of the
            # file
            if {[file exists $outpath/$indir/$outfilename/analysis_log.txt]} {
                
                # Make an initial pass through the file to determine the "lowest_collapsed_sat1" and
                # "highest_uncollapsed_sat1" values and the final stage
                set analysis_log_file [open $outpath/$indir/$outfilename/analysis_log.txt r]
                while {[gets $analysis_log_file line] >= 0} {
                
                    # Parse the contents of each line
                    set highest_uncollapsed_sat1($gm) [lindex $line 2]
                    set lowest_collapsed_sat1($gm) [lindex $line 3]
                    set current_stage($gm) [lindex $line 5]
                }
                close $analysis_log_file

                # If the bracketing stage has begun, enqueue the bracketing stage jobs
                if {$current_stage($gm) > 0} {
                    set hunt_lowest_collapsed_sat1 [expr {ceil($lowest_collapsed_sat1($gm)/$delta_sat1)* \
                            $delta_sat1}]
                    set hunt_highest_uncollapsed_sat1 [expr {floor($highest_uncollapsed_sat1($gm)/ \
                            $delta_sat1)*$delta_sat1}]

                    set bracket_intensity [expr {$hunt_lowest_collapsed_sat1 - $delta_sat1/2.0}]
                    set runlist [linsert $runlist 0 "$gm $bracket_intensity 1"]
                    dict set bracket_analyses $gm 1 $bracket_intensity

                    for {set i 2} {$i <= $num_bracket_stages} {incr i} {
                        for {set bracket_factor [expr {2.0**(-$i)}]} {$bracket_factor < 1.0} \
                                {set bracket_factor [expr {$bracket_factor + 2.0**(1 - $i)}]} {
                            set bracket_intensity [expr {$hunt_lowest_collapsed_sat1 - \
                                    $delta_sat1*(1 - $bracket_factor)}]
                            lappend runlist "$gm $bracket_intensity $i"
                            dict with bracket_analyses {
                                dict lappend $gm $i $bracket_intensity
                            }
                        }
                    }
                }

                # Loop through the contents of the "analysis_log.txt" file again to populate the "ida_curves"
                # dictionary and remove completed jobs from the "runlist" queue
                set analysis_log_file [open $outpath/$indir/$outfilename/analysis_log.txt r]
                while {[gets $analysis_log_file line] >= 0} {
                    
                    # Parse the contents of each line
                    set sat1 [lindex $line 0]
                    set drift [lindex $line 1]

                    # Update the "ida_curves" dictionary
                    if {$drift < $col_drift} {
                        dict set ida_curves $gm $sat1 $drift
                    }

                    # Remove the corresponding job from the "runlist" queue
                    for {set i 0} {$i < [llength $runlist]} {incr i} {
                        set job [lindex $runlist $i]
                        set job_gm [lindex $job 0]
                        set job_sat1 [lindex $job 1]
                        set job_stage [lindex $job 2]
                        if {($job_gm == $gm) && ($job_sat1 == $sat1)} {
                            set runlist [lreplace $runlist $i $i]
                        }
                    }
                }
                close $analysis_log_file

                # Trim the runlist queue based on the "lowest_collapsed_sat1" and "highest_uncollapsed_sat1"
                # values. This is mostly useful only if the bracketing stage has begun.
                for {set i 0} {$i < [llength $runlist]} {incr i} {
                    set job [lindex $runlist $i]
                    set job_gm [lindex $job 0]
                    set job_sat1 [lindex $job 1]
                    set job_stage [lindex $job 2]
                    if {($job_gm == $gm) && (($job_sat1 > $lowest_collapsed_sat1($gm)) || \
                            ($job_sat1 < $highest_uncollapsed_sat1($gm)))} {
                        set runlist [lreplace $runlist $i $i]
                    }
                }

                # If the ground motion has completed, display a message that says so
                if {$current_stage($gm) == -1} {
                    display_completed $gm
                }
            }

            # If previous recorder data exists, delete it
            set outdir $outpath/$indir/$outfilename
            set sat1_dirs [lsort [glob -nocomplain -directory $outdir -type d *]]
            foreach sat1_dir $sat1_dirs {
                file delete -force $sat1_dir
            }
        }
    }

    # "num_complete" stores the number of ground motions whose hunt and bracket stages have been completed
    set num_complete 0
    for {set gm 0} {$gm < $numgms} {incr gm} {
        if {$current_stage($gm) == -1} {
            incr num_complete
        }
    }

    # "free_procs" is a stack containing the numbers of all the free processes at any given point of time.
    # It is initialized to contain the ids of all the slave processes.
    for {set procid 1} {$procid < $numprocs} {incr procid} {
        lappend free_procs $procid
    }

    # If the "runlist" queue is empty, display a message that all ground motions have been completed, write
    # the "ida_curve.txt" files and terminate execution. Else, loop until all ground motions have been
    # completed, the "runlist" queue is empty, and all processes have finished execution.
    if {![info exists runlist]} {
        puts "All ground motions have been completed."
        terminate_slave_procs $numprocs
    } else {
        while {($num_complete < $numgms) || [llength $runlist] || ([llength $free_procs] < $numprocs - 1)} {
            
            # Assign jobs to all free processes by dequeueing ground motions and intensities from the
            # "runlist" queue and popping free processes from the "free_procs" stack
            while {[llength $free_procs] && [llength $runlist]} {
                
                # Dequeue a ground motion and intensity from the "runlist" queue
                set gm [lindex [lindex $runlist 0] 0]
                set sat1 [lindex [lindex $runlist 0] 1]
                set stage [lindex [lindex $runlist 0] 2]
                set runlist [lreplace $runlist 0 0]

                # Pop a free process from the "free_procs" stack
                set slave_procid [lindex $free_procs end]
                set free_procs [lreplace $free_procs end end]

                # Assign the job to the free process
                send -pid $slave_procid "$gm $sat1 $stage [dict get $gminfo_dict $gm]"
            }

            # Receive a message from a slave process and parse it to check if it is a status check or the
            # results of a completed analysis. If it is a status check, check whether the intensity is within
            # the intensity bounds and send a message to abort if it is not. If the message indicates a
            # finished analysis, push the id of the process to the "free_procs" stack and perform subsequent
            # tasks depending on whether the analysis successfully completed or aborted in between.
            set status_check_flag true
            set aborted_flag false

            while {$status_check_flag} {
                recv -pid ANY response

                set slave_procid [lindex $response 0]
                set gm [lindex $response 1]
                set sat1 [lindex $response 2]
                set stage [lindex $response 3]
                set msg [lindex $response 4]

                if {$msg == "STATUS"} {
                    if {($sat1 > $lowest_collapsed_sat1($gm)) || ($sat1 < $highest_uncollapsed_sat1($gm))} {
                        send -pid $slave_procid "ABORT"
                    } else {
                        send -pid $slave_procid "CONTINUE"
                    }
                } elseif {$msg == "ABORTED"} {
                    lappend free_procs $slave_procid
                    set status_check_flag false
                    set aborted_flag true
                } else {
                    lappend free_procs $slave_procid
                    set drift $msg
                    set status_check_flag false
                }
            }

            # If the analysis did not cause collapse, add the entry to the "ida_curves" dictionary. Note that
            # resurrected ground motion intensities could be present in the "ida_curves" dictionary.
            # Else update the "lowest_collapsed_sat1" entry for the ground motion if necessary and dequeue
            # jobs corresponding to higher intensities of this ground motion in the "runlist" queue.
            if {!$aborted_flag} {
                if {$drift < $col_drift} {
                    dict set ida_curves $gm $sat1 $drift
                } else {
                    set lowest_collapsed_sat1($gm) [expr {min($lowest_collapsed_sat1($gm), $sat1)}]

                    for {set i 0} {$i < [llength $runlist]} {incr i} {
                        set job [lindex $runlist $i]
                        set job_gm [lindex $job 0]
                        set job_sat1 [lindex $job 1]
                        set job_stage [lindex $job 2]
                        if {($job_gm == $gm) && ($job_sat1 > $lowest_collapsed_sat1($gm))} {
                            set runlist [lreplace $runlist $i $i]
                        }
                    }
                }
            }

            # If the hunt stage is complete, modify "current_stage" and enqueue the bracket stage jobs such
            # that the bracket stage 1 job is at the head of the queue and higher stage jobs are at the tail
            # of the queue. Also store the enqueued bracket stage jobs in the "bracket_analyses" dictionary.
            if {!$aborted_flag && ($stage == 0) && ($sat1 <= $lowest_collapsed_sat1($gm)) && \
                    ($lowest_collapsed_sat1($gm) != inf)} {

                # Count the uncollapsed intensities including zero
                set num_uncollapsed_analyses 0
                foreach uncollapsed_sat1 [dict keys [dict get $ida_curves $gm]] {
                    if {$uncollapsed_sat1 < $lowest_collapsed_sat1($gm)} {
                        incr num_uncollapsed_analyses
                    }
                }

                # Compute the required number of uncollapsed intensities including zero
                set num_uncollapsed_analyses_req [expr {round($lowest_collapsed_sat1($gm)/$delta_sat1)}]

                # Enqueue the bracketing stage jobs and update the "highest_uncollapsed_sat1" entry
                # corresponding to the ground motion if the hunt stage is complete
                if {$num_uncollapsed_analyses == $num_uncollapsed_analyses_req} {
                    set current_stage($gm) 1
                    set highest_uncollapsed_sat1($gm) [expr {$lowest_collapsed_sat1($gm) - $delta_sat1}]

                    set bracket_intensity [expr {$lowest_collapsed_sat1($gm) - $delta_sat1/2.0}]
                    set runlist [linsert $runlist 0 "$gm $bracket_intensity 1"]
                    dict set bracket_analyses $gm 1 $bracket_intensity

                    for {set i 2} {$i <= $num_bracket_stages} {incr i} {
                        for {set bracket_factor [expr {2.0**(-$i)}]} {$bracket_factor < 1.0} \
                                {set bracket_factor [expr {$bracket_factor + 2.0**(1 - $i)}]} {
                            set bracket_intensity [expr {$lowest_collapsed_sat1($gm) - \
                                    $delta_sat1*(1 - $bracket_factor)}]
                            lappend runlist "$gm $bracket_intensity $i"
                            dict with bracket_analyses {
                                dict lappend $gm $i $bracket_intensity
                            }
                        }
                    }
                }
            }
            
            # Tasks to perform if the analysis is from the bracketing stage
            if {$stage > 0} {
                
                # If the analysis did not cause collapse, update the "highest_uncollapsed_sat1" entry for
                # the ground motion if necessary and dequeue jobs corresponding to lower intensities in the
                # "runlist" queue.
                if {!$aborted_flag && ($drift < $col_drift)} {
                    set highest_uncollapsed_sat1($gm) [expr {max($highest_uncollapsed_sat1($gm), $sat1)}]

                    for {set i 0} {$i < [llength $runlist]} {incr i} {
                        set job [lindex $runlist $i]
                        set job_gm [lindex $job 0]
                        set job_sat1 [lindex $job 1]
                        set job_stage [lindex $job 2]
                        if {($job_gm == $gm) && ($job_sat1 < $highest_uncollapsed_sat1($gm))} {
                            set runlist [lreplace $runlist $i $i]
                        }
                    }
                }

                # Determine if the current stage has completed by checking if any bracketing stage analyses
                # were spawned, that lie within the intensity bounds. Update the "current_stage" of the
                # ground motion iteratively if required. If the ground motion has completed, mark it as so and
                # display a message.
                set stage_update_flag false
                if {$stage == $current_stage($gm)} {
                    while {true} {
                        set stage_update_flag true
                        foreach bracket_sat1 [dict get $bracket_analyses $gm $current_stage($gm)] {
                            if {($bracket_sat1 < $lowest_collapsed_sat1($gm)) && \
                                    ($bracket_sat1 > $highest_uncollapsed_sat1($gm))} {
                                set stage_update_flag false
                                break
                            }
                        }
                        if {$stage_update_flag} {
                            if {$current_stage($gm) == $num_bracket_stages} {
                                set current_stage($gm) -1
                                display_completed $gm
                                break
                            } else {
                                incr current_stage($gm)
                            }
                        } else {
                            break
                        }
                    }
                }

                # If the bracketing stage has been updated, move the job entry in "runlist" corresponding to
                # the next bracket stage intensity to the head of the queue if not running already and job
                # entires corresponding to later bracket stages to the tail of the queue if not running
                # already.
                if {$stage_update_flag} {
                    set new_runlist_head ""
                    set new_runlist_body ""
                    set new_runlist_tail ""

                    for {set i 0} {$i < [llength $runlist]} {incr i} {
                        set job [lindex $runlist $i]
                        set job_gm [lindex $job 0]
                        set job_sat1 [lindex $job 1]
                        set job_stage [lindex $job 2]
                        if {($job_gm == $gm) && ($job_stage == $current_stage($gm))} {
                            lappend new_runlist_head $job
                        } elseif {($job_gm == $gm) && ($job_stage > $current_stage($gm))} {
                            lappend new_runlist_tail $job
                        } else {
                            lappend new_runlist_body $job
                        }
                    }

                    set runlist [concat $new_runlist_head $new_runlist_body $new_runlist_tail]
                }
            }

            # Write the result to the "analysis_log.txt" file in the output folder corresponding
            # to the current ground motion
            if {!$aborted_flag} {
                set gminfo [dict get $gminfo_dict $gm]
                dict with gminfo {
                    set analysis_log_file [open $outpath/$indir/$outfilename/analysis_log.txt a]
                    puts $analysis_log_file "[format "%.5f" $sat1] [format "%.5f" $drift]\
                            [format "%.5f" $highest_uncollapsed_sat1($gm)]\
                            [format "%.5f" $lowest_collapsed_sat1($gm)] $stage $current_stage($gm)"
                    close $analysis_log_file
                }
            }

            # Compute the total number of ground motions whose hunt and bracket stages have been completed
            set num_complete 0
            for {set gm 0} {$gm < $numgms} {incr gm} {
                if {$current_stage($gm) == -1} {
                    incr num_complete
                }
            }
        }

        # Send a message to all slave processes to terminate execution
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
                # than the "lowest_collapsed_sat1" for the ground motion
                if {$sat1 < $lowest_collapsed_sat1($gm)} {
                    puts $ida_curve_file "[format "%.5f" $sat1] [format "%.5f" $drift]"
                }
            }
            close $ida_curve_file
        }
    }

# Tasks to be executed by all slave processes
} else {

    # Source the required files
    source constants_units_kip_in.tcl
    source $modeldir/frame_data.tcl
    source create_steel_mf_model.tcl
    source max_drift_check.tcl

    # Define the list of nodes used to compute story drifts
    set ctrl_nodes 1[format "%02d" [expr {$num_bays + 1}]]002
    for {set story 1} {$story <= $num_stories} {incr story} {
        lappend ctrl_nodes 1[format "%02d" [expr {$num_bays + 1}]][format "%02d" $story]4
    }

    # Receive a message from the master process containing the description of a job, run the job, report back
    # to the master process, and wait for a new message
    recv -pid 0 command

    for {set i 0} {$command != "DONE"} {incr i} {

        # Parse the command from the master process
        set gm [lindex $command 0]
        set sat1 [lindex $command 1]
        set stage [lindex $command 2]
        set gminfo [lrange $command 3 end]

        # Wipe previous model definitions, create the model, and run the analysis
        wipe
        CreateSteelMFModel $modeldir/frame_data.tcl
        dict with gminfo {
            source recorders_analysis_ida_mp_hpc.tcl
        }

        # Send the master node the results of the analysis and receive a new command
        if {$abort_flag} {
            send -pid 0 "$this_procid $gm $sat1 $stage ABORTED"
        } else {
            send -pid 0 "$this_procid $gm $sat1 $stage $max_drift"
        }
        recv -pid 0 command
    }
