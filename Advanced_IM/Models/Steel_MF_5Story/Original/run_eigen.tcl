##############################################################################################################
# Reagan Chandramohan                                                                                        #
# John A. Blume Earthquake Engineering Center                                                                #
# Stanford University                                                                                        #
# Last edited: 02-Mar-2016                                                                                   #
##############################################################################################################

# Run an eigenvalue analysis
# Write the mode shapes, periods, and node and element information to their respective text files.

##############################################################################################################

# Wipe previous model definitions and build the model
wipe
source Model.tcl

# Define the output directory and mode-shape recorders for all modes
set recorderdir Model_Info
file mkdir $recorderdir
recorder Node -file $recorderdir/mode_shapes.out -dof 1 2 3 eigen

# Define the number of dof in the model
set num_dof 232

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
