##############################################################################################################
# Reagan Chandramohan                                                                                        #
# John A. Blume Earthquake Engineering Center                                                                #
# Stanford University                                                                                        #
# Last edited: 07-Dec-2014                                                                                   #
##############################################################################################################

# Run an eigenvalue analysis on all the RC moment frame models.
# Write the mode shapes, periods, and node and element information to their respective text files.

##############################################################################################################

# Loop over all the model directories
set modelpath Models
set modeldir $modelpath/BRBF_12Story

# Wipe previous model definitions and build the model
wipe
source $modeldir/build_modified_atc84_model.tcl

# Define the output directory and mode-shape recorders for all modes
set recorderdir $modeldir/Model_Info
file mkdir $recorderdir
recorder Node -file $recorderdir/mode_shapes.out -node 317 326 336 346 356 366 376 386 396 3106 3116 3126 3136 -dof 1 eigen 

# Compute the number of dof in the model
set slash_loc [string first / $modeldir]
set model_type [string range $modeldir [expr {$slash_loc + 1}] [expr {$slash_loc + 4}]]
if {$model_type == "BRBF"} {
    set num_dof [expr {95*($num_stories/2) + 49*($num_stories%2)}]
} else {
    set num_dof [expr {227*($num_stories/2) + 115*($num_stories%2)}]
}

# Run the eigenvalue analysis and compute all modal periods
# fullGenLapack for all except BRBF_12Story
# use $num_dof for all except BRBF_12Story which is ($num_dof-1)  
# set eigenvalues [eigen -fullGenLapack $num_dof]
set eigenvalues [eigen [expr $num_dof-1]]
puts $eigenvalues
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
