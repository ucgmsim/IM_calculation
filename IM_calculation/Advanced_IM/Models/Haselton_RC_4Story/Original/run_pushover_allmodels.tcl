##############################################################################################################
# Reagan Chandramohan                                                                                        #
# John A. Blume Earthquake Engineering Center                                                                #
# Stanford University                                                                                        #
# Last edited: 13-Oct-2014                                                                                   #
##############################################################################################################

# Run a pushover analysis on all the RC moment frame models

##############################################################################################################

# Run one pushover analyis step
# Return "true" if analysis is not yet completed
# Return "false" if the current base shear is less than 40% the peak base shear, indicating the analysis has
# completed
proc RunPushoverStep {} {
    
    # Reference variables from higher scope
    upvar unloading unloading
    upvar cur_base_shear cur_base_shear
    upvar peak_base_shear peak_base_shear

    # Run one pushover analyis step and return "false" if the step did not complete successfully
    if {[analyze 1]} {
        puts "Analysis Error"
        return false
    }

    # Get the current base shear and check if it's less than the peak base shear. If it is, note that the
    # structure is unloading. If it is not, and the structure is still loading, update the peak base shear.
    set cur_base_shear [getTime]
    if {$cur_base_shear < $peak_base_shear} {
        set unloading true
    } elseif {!$unloading || $cur_base_shear > $peak_base_shear} {
        set peak_base_shear $cur_base_shear
    }

    # If the structure is unloading and the current base shear is less than 40% the peak base shear, return
    # "false"
    if {$unloading && $cur_base_shear < 0.4*$peak_base_shear} {
        puts "Complete"
        return false
    }

    return true
}

# Set the number of steps to run the analysis in
#set numsteps 50000
set numsteps 60000

# Loop over all the model directories
set modelpath Models
set modeldirs [lsort [glob -directory $modelpath -type d *]]
#set modeldirs "Models/ID1020_20Story"
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
    source create_model.tcl

    # Define the story and roof drift recorders
    for {set story 1} {$story <= $num_stories} {incr story} {
        recorder Drift -file $recorderdir/story${story}_drift.out -time -iNode [lindex $ctrl_nodes \
                [expr {$story - 1}]] -jNode [lindex $ctrl_nodes $story] -dof 1 -perpDirn 2
    }
    set roof_node [lindex $ctrl_nodes end]
    recorder Drift -file $recorderdir/roof_drift.out -time -iNode [lindex $ctrl_nodes 0] -jNode \
            $roof_node -dof 1 -perpDirn 2

    # Define the pushover load pattern such that the sum of all lateral loads equals 1. This allows
    # interpretation of the time step of the analysis as the applied load ratio.
    source $modeldir/define_haselton_pushover_load.tcl

    # Compute the target roof displacement
    set target_roof_drift 0.20
    set roof_height [nodeCoord $roof_node 2]
    set target_roof_disp [expr {$target_roof_drift*$roof_height}]

    # Initialize the analysis parameters
    constraints Transformation
    numberer RCM
    system SparseGEN
    algorithm Linear
    set anal_ctrl_node 3[format "%02d" [expr {$num_stories + 1}]]1
    integrator DisplacementControl $anal_ctrl_node 1 [expr {$target_roof_disp/$numsteps}]
    analysis Static

    # Run the analysis step-by-step until the base shear is less than 40% the peak base shear
    set unloading false
    set peak_base_shear 0.0
    set cur_base_shear 0.0

    for {set step 1} {$step <= $numsteps} {incr step} {
        if {![RunPushoverStep]} {
            break
        }
    }
}
