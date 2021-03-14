##############################################################################################################
# Reagan Chandramohan                                                                                        #
# John A. Blume Earthquake Engineering Center                                                                #
# Stanford University                                                                                        #
# Last edited: 03-Mar-2016                                                                                   #
##############################################################################################################

# Run a pushover analysis on all the SAC steel moment frame models

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

    # Run one pushover analyis step
    # If the step fails, decrease the step size by 50%
    # After 20 attempts, return "false" to indicate that the analysis failed
    while {[analyze 1]} {
        incr num_attempts
        puts "Attempt $num_attempts"
        set numsteps [expr {$numsteps*2}]
        set step_size [expr {$step_size/2.0}]

        constraints Transformation
        numberer RCM
        system SparseGEN
        algorithm Linear
        integrator DisplacementControl $anal_ctrl_node 1 $step_size
        analysis Static

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
set numsteps 20000

# Loop over all the model directories
set modelpath Models
set modeldirs [lsort [glob -directory $modelpath -type d *Story]]
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
    source constants_units_kip_in.tcl
    source $modeldir/frame_data.tcl
    source create_steel_mf_model.tcl
    CreateSteelMFModel $modeldir/frame_data.tcl

    # Define the story drift recorders at the leaning column
    set lc_bay [expr {$num_bays + 1}]
    for {set story [expr {$num_basement_levels + 1}]} {$story <= $num_stories} {incr story} {
        if {$story == 1} {
            set nodei 1[format "%02d" $lc_bay]002
        } else {
            set nodei 1[format "%02d" $lc_bay][format "%02d" [expr {$story - 1}]]4
        }
        set nodej 1[format "%02d" $lc_bay][format "%02d" $story]4
        recorder Drift -file $recorderdir/story${story}_drift.out -time -iNode $nodei -jNode $nodej -dof 1 \
                -perpDirn 2
    }

    # Define the roof drift recorder at the leaning column
    if {$basement_flag} {
        set ground_node 1[format "%02d" $lc_bay][format "%02d" $num_basement_levels]4
    } else {
        set ground_node 1[format "%02d" $lc_bay]002
    }
    set roof_node 1[format "%02d" $lc_bay][format "%02d" $num_stories]4
    recorder Drift -file $recorderdir/roof_drift.out -time -iNode $ground_node -jNode \
        $roof_node -dof 1 perpDirn 2

#    # Compute the product of the mass matrix and first mode shape to be used as the lateral load pattern
#    # Maintain a sum of the displacements at all stories.
#    set load_profile_sum 0.0
#    for {set story [expr {$num_basement_levels + 1}]} {$story <= $num_stories} {incr story} {
#        set story_mass [lindex $story_masses [expr {$story - $num_basement_levels - 1}]]
#        set story_disp [nodeEigenvector 100[format "%02d" $story]3 1 1]
#        set load_profile($story) [expr {$story_mass*$story_disp}]
#        set load_profile_sum [expr {$load_profile_sum + $load_profile($story)}]
#    }

    # Compute the first mode shape to be used as the lateral load pattern
    # Maintain the sum of the displacements at all stories
    set load_profile_sum 0.0
    for {set story [expr {$num_basement_levels + 1}]} {$story <= $num_stories} {incr story} {
        set load_profile($story) [nodeEigenvector 100[format "%02d" $story]3 1 1]
        set load_profile_sum [expr {$load_profile_sum + $load_profile($story)}]
    }

    # Define the pushover load pattern such that the sum of all lateral loads equals 1. This allows
    # interpretation of the time step of the analysis as the applied load ratio.
    set pushover_load_ts 2
    timeSeries Linear $pushover_load_ts

    set pushover_load_pattern 2
    pattern Plain $pushover_load_pattern $pushover_load_ts {
        for {set story [expr {$num_basement_levels + 1}]} {$story <= $num_stories} {incr story} {
            for {set bay 0} {$bay <= $num_bays} {incr bay} {
                load 1[format "%02d" $bay][format "%02d" $story]3 \
                        [expr {$load_profile($story)/$load_profile_sum/($num_bays + 1)}] 0.0 0.0
            }
        }
    }

    # Compute the target roof displacement
    set target_roof_drift 0.20
    set roof_height [nodeCoord $roof_node 2]
    set target_roof_disp [expr {$target_roof_drift*$roof_height}]
    set step_size [expr {$target_roof_disp/$numsteps}]

    # Initialize the analysis parameters
    constraints Transformation
    numberer RCM
    system SparseGEN
    algorithm Linear
    set anal_ctrl_node $roof_node
    integrator DisplacementControl $anal_ctrl_node 1 $step_size
    analysis Static

    # Run the analysis step-by-step until the base shear is less than 70% the peak base shear
    set unloading false
    set peak_base_shear 0.0
    set cur_base_shear 0.0
    set num_attempts 0

    while {$numsteps > 0 && [RunPushoverStep]} {
        incr numsteps -1
    }
}
