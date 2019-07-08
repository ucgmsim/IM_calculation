##############################################################################################################
# Reagan Chandramohan                                                                                        #
# John A. Blume Earthquake Engineering Center                                                                #
# Stanford University                                                                                        #
# Last edited: 02-Mar-2016                                                                                   #
##############################################################################################################

# Run an eigenvalue analysis on all the SAC steel moment frame models.
# Write the mode shapes, periods, and node and element information to their respective text files.

##############################################################################################################

# Loop over all the model directories
set modelpath Models
set modeldirs [lsort [glob -directory $modelpath -type d *Story]]
foreach modeldir $modeldirs {

    # Wipe previous model definitions and build the model
    wipe
    source constants_units_kip_in.tcl
    source $modeldir/frame_data.tcl
    source create_steel_mf_model.tcl
    CreateSteelMFModel $modeldir/frame_data.tcl
  
    # Display the model name
    puts \n$modeldir

    # Define the output directory and mode-shape recorders for all modes
    set recorderdir $modeldir/Model_Info
    file mkdir $recorderdir
    recorder Node -file $recorderdir/mode_shapes.out -dof 1 2 3 eigen

    # Compute the number of dof in the model
    set slash_loc [string first / $modeldir]
    set model_name [string range $modeldir [expr {$slash_loc + 1}] end]
    if {[string compare $model_name "3Story"] == 0} {
        set num_dof 213
    } elseif {[string compare $model_name "9Story"] == 0} {
        set num_dof 699
    }

    # Run the eigenvalue analysis and compute all modal periods
    set eigenvalues [eigen -fullGenLapack $num_dof]
    set periods {}
    foreach eigenvalue $eigenvalues {
        lappend periods [expr {2.0*$pi/sqrt($eigenvalue)}]
    }

    # Record the mode shapes into the recorders
    record 

    # Save the modal periods in an output file
    set period_file [open $recorderdir/periods.out w]
    foreach T $periods {
        puts $period_file $T
    }
    close $period_file

    # Display the fundamental period and lowest period
    puts "Fundamental mode period: [format "%.4f" [lindex $periods 0]] s"
    puts "Lowest modal period: [format "%.4e" [lindex $periods end]] s"

    # Store details about all nodes in an output file (delete the file if it already exists before creating a
    # new one since otherwise data is appended to the files by OpenSees)
    file delete $recorderdir/node_info.out
    file delete $recorderdir/element_info.out

    print $recorderdir/node_info.out -node
    print $recorderdir/element_info.out -ele
}
