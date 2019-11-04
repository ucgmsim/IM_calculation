# 
# define folders for recorders
    #  file mkdir $Output_path 
    # file mkdir $Output_path/drift
	file mkdir $Output_path/env_drift
	file mkdir $Output_path/env_disp
	file mkdir $Output_path/env_accl 

#
# --Define Time Series--
#

source [file join [file dirname [info script]] ../general/time_series.tcl]
	
# Define Recorders
#
# --------------- Drift ----------------------

# --Define the story and roof drift recorders--

# for {set story 1} {$story <= $num_stories} {incr story} {
    # recorder Drift -file $Output_path/drift/story${story}_drift.out -time -iNode [lindex $ctrl_nodes \
            # [expr {$story - 1}]] -jNode [lindex $ctrl_nodes $story] -dof 1 -perpDirn 2
# }
# set roof_node [lindex $ctrl_nodes end]
# recorder Drift -file $Output_path/drift/roof_drift.out -time -iNode [lindex $ctrl_nodes 0] -jNode \
        # $roof_node -dof 1 -perpDirn 2

		
# --Define the story and roof envelope drift recorders--

for {set story 1} {$story <= $num_stories} {incr story} {
    set recTags($story) [recorder EnvelopeDrift -file $Output_path/env_drift/drift_story${story}.out -time -iNode [lindex $ctrl_nodes \
            [expr {$story - 1}]] -jNode [lindex $ctrl_nodes $story] -dof 1 -perpDirn 2]
}
# set roof_node [lindex $ctrl_nodes end]
# recorder EnvelopeDrift -file $Output_path/env_drift/drift_roof.out -time -iNode [lindex $ctrl_nodes 0] -jNode \
        # $roof_node -dof 1 -perpDirn 2

# --------------- Disp ----------------------

for {set story 0} {$story <= $num_stories} {incr story} {
    recorder EnvelopeNode -file $Output_path/env_disp/disp_story${story}.out -time -node [lindex $ctrl_nodes $story] -dof 1 disp
}
# set roof_node [lindex $ctrl_nodes end]
# recorder EnvelopeNode -file $Output_path/env_disp/disp_roof.out -time -node $roof_node -dof 1 disp

# --------------- Acc ----------------------

for {set story 0} {$story <= $num_stories} {incr story} {
    recorder EnvelopeNode -file $Output_path/env_accl/accl_story${story}.out -time -timeSeries 5 -node [lindex $ctrl_nodes $story] -dof 1 accel
	# set nn [lindex $ctrl_nodes $story] 
   	# puts "node= $nn"
}
# set roof_node [lindex $ctrl_nodes end]
# recorder EnvelopeNode -file $Output_path/env_accl/accl_roof.out -time -timeSeries 5 -node $roof_node -dof 1 accel

# ----------------------------------------------
puts "Recorders are defined!"
# ----------------------------------------------


