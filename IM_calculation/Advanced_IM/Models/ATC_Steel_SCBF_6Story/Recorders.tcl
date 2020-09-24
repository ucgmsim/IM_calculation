# 
# define folders for recorders
        file mkdir $Output_path/env_drift
	file mkdir $Output_path/env_disp
	file mkdir $Output_path/env_accl 
	# file mkdir $Output_path/drift
	# file mkdir $Output_path/accl
         # file mkdir $Output_path/disp
#
# --Define Time Series--
#
source [file join [file dirname [info script]] ../general/time_series.tcl]
	
# Define Recorders
#
# --------------- Drift ----------------------

# --Define the story and roof drift recorders--
# time-series
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
# recorder EnvelopeDrift -file $Output_path/env_drift/roof_drift_env.out -time -iNode [lindex $ctrl_nodes 0] -jNode \
        # $roof_node -dof 1 -perpDirn 2

# --------------- Disp ----------------------
# time-series
# for {set story 1} {$story <= $num_stories} {incr story} {
    # recorder Node -file $Output_path/disp/story${story}_disp.out -time  -node [lindex $ctrl_nodes $story] -dof 1  disp
# }

# Envelope
for {set story 0} {$story <= $num_stories} {incr story} {
    recorder EnvelopeNode -file $Output_path/env_disp/disp_story${story}.out -time -node [lindex $ctrl_nodes $story] -dof 1 disp
}
# set roof_node [lindex $ctrl_nodes end]
# recorder EnvelopeNode -file $Output_path/disp/roof_disp_env.out -time -node $roof_node -dof 1 disp

# --------------- Acc ----------------------
# time-series
# for {set story 0} {$story <= $num_stories} {incr story} {
    # recorder Node -file $Output_path/accl/story${story}_accl.out -time -timeSeries 5 -node [lindex $ctrl_nodes $story] -dof 1  accel 
# }

# Envelope
for {set story 0} {$story <= $num_stories} {incr story} {
    recorder EnvelopeNode -file $Output_path/env_accl/accl_story${story}.out -time -timeSeries 5 -node [lindex $ctrl_nodes $story] -dof 1 accel
	# set nn [lindex $ctrl_nodes $story] 
   	# puts "node= $nn"
}
# set roof_node [lindex $ctrl_nodes end]
# recorder EnvelopeNode -file $Output_path/env_accl/roof_accl_env.out -time -timeSeries 5 -node $roof_node -dof 1 accel

# ----------------------------------------------
puts "Recorders are defined!"
# ----------------------------------------------


