# 
# define folders for recorders
        file mkdir ../Outputs/$GM_name/$model_name/$station_name
	file mkdir ../Outputs/$GM_name/$model_name/$station_name/drift
	file mkdir ../Outputs/$GM_name/$model_name/$station_name/env_drift
	file mkdir ../Outputs/$GM_name/$model_name/$station_name/env_disp
	file mkdir ../Outputs/$GM_name/$model_name/$station_name/env_accl 

#
# --Define Time Series--
#
source Models/general/time_series.tcl
	
# Define Recorders
#
# --------------- Drift ----------------------

# --Define the story and roof drift recorders--

# for {set story 1} {$story <= $num_stories} {incr story} {
    # recorder Drift -file ../Outputs/$GM_name/$model_name/$type_GM/$station_name/drift/story${story}_drift.out -time -iNode [lindex $ctrl_nodes \
            # [expr {$story - 1}]] -jNode [lindex $ctrl_nodes $story] -dof 1 -perpDirn 2
# }
# set roof_node [lindex $ctrl_nodes end]
# recorder Drift -file ../Outputs/$GM_name/$model_name/$type_GM/$station_name/drift/roof_drift.out -time -iNode [lindex $ctrl_nodes 0] -jNode \
        # $roof_node -dof 1 -perpDirn 2

		
# --Define the story and roof envelope drift recorders--

for {set story 1} {$story <= $num_stories} {incr story} {
    set recTags($story) [recorder EnvelopeDrift -file ../Outputs/$GM_name/$model_name/$station_name/env_drift/story${story}_drift_env.out -time -iNode [lindex $ctrl_nodes \
            [expr {$story - 1}]] -jNode [lindex $ctrl_nodes $story] -dof 1 -perpDirn 2]
}
set roof_node [lindex $ctrl_nodes end]
recorder EnvelopeDrift -file ../Outputs/$GM_name/$model_name/$station_name/env_drift/roof_drift_env.out -time -iNode [lindex $ctrl_nodes 0] -jNode \
        $roof_node -dof 1 -perpDirn 2

# --------------- Disp ----------------------

for {set story 1} {$story <= $num_stories} {incr story} {
    recorder EnvelopeNode -file ../Outputs/$GM_name/$model_name/$station_name/env_disp/story${story}_disp_env.out -time -node [lindex $ctrl_nodes $story] -dof 1 disp
}
set roof_node [lindex $ctrl_nodes end]
recorder EnvelopeNode -file ../Outputs/$GM_name/$model_name/$station_name/env_disp/roof_disp_env.out -time -node $roof_node -dof 1 disp

# --------------- Acc ----------------------

for {set story 1} {$story <= $num_stories} {incr story} {
    recorder EnvelopeNode -file ../Outputs/$GM_name/$model_name/$station_name/env_accl/story${story}_accl_env.out -time -timeSeries 5 -node [lindex $ctrl_nodes $story] -dof 1 accel
	# set nn [lindex $ctrl_nodes $story] 
   	# puts "node= $nn"
}
set roof_node [lindex $ctrl_nodes end]
recorder EnvelopeNode -file ../Outputs/$GM_name/$model_name/$station_name/env_accl/roof_accl_env.out -time -timeSeries 5 -node $roof_node -dof 1 accel

# ----------------------------------------------
puts "Recorders are defined!"
# ----------------------------------------------


