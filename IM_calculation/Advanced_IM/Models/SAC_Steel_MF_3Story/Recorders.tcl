# 
# define folders for recorders
file mkdir $Output_path/env_drift
file mkdir $Output_path/env_disp
file mkdir $Output_path/env_accl 
#
# --------------------------------------------  
# Define the list of nodes used to compute story drifts
set ctrl_nodes 1[format "%02d" [expr {$num_bays + 1}]]002
for {set story 1} {$story <= $num_stories} {incr story} {
    lappend ctrl_nodes 1[format "%02d" [expr {$num_bays + 1}]][format "%02d" $story]4
}   

# --Define Time Series--
#

source [file join [file dirname [info script]] ../general/time_series.tcl]
    
# Define Recorders
#
# --------------- Drift ----------------------
# --Define the story and roof envelope drift recorders--

for {set story 1} {$story <= $num_stories} {incr story} {
    set recTags($story) [recorder EnvelopeDrift -file $Output_path/env_drift/drift_story${story}.out -time -iNode [lindex $ctrl_nodes \
            [expr {$story - 1}]] -jNode [lindex $ctrl_nodes $story] -dof 1 -perpDirn 2]
}

# --------------- Disp ----------------------
for {set story 0} {$story <= $num_stories} {incr story} {
    recorder EnvelopeNode -file $Output_path/env_disp/disp_story${story}.out -time -node [lindex $ctrl_nodes $story] -dof 1 disp
}

# --------------- Acc ----------------------
# Envelope
for {set story 0} {$story <= $num_stories} {incr story} {
    recorder EnvelopeNode -file $Output_path/env_accl/accl_story${story}.out -time -timeSeries 5 -node [lindex $ctrl_nodes $story] -dof 1 accel
}

# ----------------------------------------------
puts "Recorders are defined!"
# ----------------------------------------------


