##############################################################################################################
# Reagan Chandramohan                                                                                        #
# John A. Blume Earthquake Engineering Center                                                                #
# Stanford University                                                                                        #
# Last edited: 24-Aug-2015                                                                                   #
##############################################################################################################

# Run a pushover analysis on all the RC moment frame models

##############################################################################################################

# Run one pushover analyis step
# Return "true" if analysis is not yet completed
# Return "false" if the current base shear is less than 70% the peak base shear, indicating the analysis has
# completed
proc RunPushoverStep {} {
    
    # Reference variables from higher scope
    upvar unloading unloading
    upvar cur_base_shear cur_base_shear
    upvar peak_base_shear peak_base_shear
    upvar numsteps numsteps
    upvar anal_ctrl_node anal_ctrl_node
    upvar step_size step_size
    upvar num_attempts num_attempts
   	
     upvar roof_height roof_height 
     upvar target_roof_disp target_roof_disp 
     upvar target_roof_drift target_roof_drift 	 
     upvar roof_node roof_node
	 
    # Run one pushover analyis step
    # If the step fails, decrease the step size by 50%
    # After 20 attempts, return "false" to indicate that the analysis failed
    while {[analyze 1]} {
        incr num_attempts
        puts "Attempt $num_attempts"
        set numsteps [expr {$numsteps*2}]
        set step_size [expr {$step_size/2.0}]

        source analyze.pushover.run.tcl

        if {$num_attempts == 20} {
            puts "Analysis Failed"
            return false
        }
    }

    # Get the current base shear and check if it's less than the peak base shear. If it is, note that the
    # structure is unloading. If it is not, and the structure is still loading, update the peak base shear.
    set cur_base_shear [getTime]
    if {$cur_base_shear < $peak_base_shear} {
        set unloading true
    } elseif {!$unloading || $cur_base_shear > $peak_base_shear} {
        set peak_base_shear $cur_base_shear
    }

    # If the structure is unloading and the current base shear is less than 70% the peak base shear, return
    # "false"
    if {$unloading && $cur_base_shear < 0.7*$peak_base_shear} {
        puts "Complete"
        return false
    }

    return true
}

# Set the number of steps to run the analysis in
set numsteps 40000

# Loop over all the model directories
set modelpath Models
#set modeldirs [lsort [glob -directory $modelpath -type d *]]
set modeldirs "Models/SCBF_3Story"
foreach modeldir $modeldirs {

    # Display the model name
    puts \n$modeldir

    # Define the output directory 
    set slash_loc [string first / $modeldir]
    set modelname [string range $modeldir [expr {$slash_loc + 1}] end]
    set recorderdir Analysis_Results/$modelname/Pushover
    file mkdir $recorderdir

    # Wipe previous model definitions and build the model
    wipe
    source $modeldir/build_modified_atc84_model.tcl

    # Define the story and roof drift recorders
    for {set story 1} {$story <= $num_stories} {incr story} {
        recorder Drift -file $recorderdir/story${story}_drift.out -time -iNode [lindex $ctrl_nodes \
                [expr {$story - 1}]] -jNode [lindex $ctrl_nodes $story] -dof 1 -perpDirn 2
    }
    set roof_node [lindex $ctrl_nodes end]
    recorder Drift -file $recorderdir/roof_drift.out -time -iNode [lindex $ctrl_nodes 0] -jNode \
            $roof_node -dof 1 -perpDirn 2

    # Compute the first mode shape to be used as the lateral load pattern
    # Compute the sum of the displacements at all stories
    set first_mode {}
    set first_mode_sum 0.0
    for {set story 1} {$story <= $num_stories} {incr story} {
        set story_disp [nodeEigenvector [lindex $ctrl_nodes $story] 1 1]
        lappend first_mode $story_disp
        set first_mode_sum [expr {$first_mode_sum + $story_disp}]
    }

    # Define the pushover load pattern such that the sum of all lateral loads equals 1. This allows
    # interpretation of the time step of the analysis as the applied load ratio.
    pattern Plain 2 Linear {
        for {set story 1} {$story <= $num_stories} {incr story} {
            for {set bay 1} {$bay <= 2} {incr bay} {
                load $bay[expr {$story + 1}] \
                        [expr {[lindex $first_mode [expr {$story - 1}]]/$first_mode_sum/2.0}] 0.0 0.0
            }
        }
    }

    # Compute the target roof displacement
    set target_roof_drift 0.20
    set roof_height [nodeCoord $roof_node 2]
    set target_roof_disp [expr {$target_roof_drift*$roof_height}]
    set step_size [expr {$target_roof_disp/$numsteps}]
    # puts $step_size
   
	
    # Initialize the analysis parameters
  source analyze.pushover.run.tcl

    # Run the analysis step-by-step until the base shear is less than 70% the peak base shear
    set unloading false
    set peak_base_shear 0.0
    set cur_base_shear 0.0
    set num_attempts 0

    while {$numsteps > 0 && [RunPushoverStep]} {
        incr numsteps -1
    }
}
