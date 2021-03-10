##############################################################################################################
# Reagan Chandramohan                                                                                        #
# John A. Blume Earthquake Engineering Center                                                                #
# Stanford University                                                                                        #
# Last edited: 05-Oct-2014                                                                                   #
##############################################################################################################

# Compute the peak story drift in the structure during the analysis from the output envelope drift files
# Used to detect if the structure has collapsed after the analysis has completed
proc max_drift_outfile {recorderdir num_stories} {
    
    # Flush the recorder file buffers
    wipe

    # Loop over all the stories to extract the peak story drift in the structure
    set max_drift 0.0
    for {set story 1} {$story <= $num_stories} {incr story} {
        set driftfile [open $recorderdir/story${story}_drift_env.out r]

        # Read the first two lines from the file and then read the peak story drift
        for {set j 1} {$j <= 2} {incr j} {gets $driftfile line}
        gets $driftfile story_max_drift
        close $driftfile

        # Compare the peak story drift to the previous largest and update if it is larger
        if {$story_max_drift > $max_drift} {
            set max_drift $story_max_drift
        }
    }

    return $max_drift
}

# Compute the peak story drift in a moment frame model from the current nodal coordinates
# Used to detect if the structure has collapsed before the analysis has completed
proc max_drift_model {ctrl_nodes} {
    
    # Retrieve the x and y coordinates of all the control nodes
    foreach ctrl_node $ctrl_nodes {
        lappend ctrl_node_xcoords [nodeCoord $ctrl_node 1]
        lappend ctrl_node_ycoords [nodeCoord $ctrl_node 2]
    }

    # Retrieve the x and y displacements at all the control nodes
    foreach ctrl_node $ctrl_nodes {
        lappend ctrl_node_xdisps [nodeDisp $ctrl_node 1]
        lappend ctrl_node_ydisps [nodeDisp $ctrl_node 2]

        # Check if the displacements are nan
        if {([lindex $ctrl_node_xdisps end] != [lindex $ctrl_node_xdisps end]) || \
                ([lindex $ctrl_node_ydisps end] != [lindex $ctrl_node_ydisps end])} {
            return inf
        }
    }

    # Loop over all the stories to extract the peak story drift in the structure
    set max_drift 0.0
    for {set i 0} {$i < [expr {[llength $ctrl_nodes] - 1}]} {incr i} {
        
        # Compute the peak story drift
        set deltax [expr {[lindex $ctrl_node_xcoords [expr {$i + 1}]] + [lindex $ctrl_node_xdisps \
                [expr {$i + 1}]] - [lindex $ctrl_node_xcoords $i] - [lindex $ctrl_node_xdisps $i]}]
        set deltay [expr {[lindex $ctrl_node_ycoords [expr {$i + 1}]] + [lindex $ctrl_node_ydisps \
                [expr {$i + 1}]] - [lindex $ctrl_node_ycoords $i] - [lindex $ctrl_node_ydisps $i]}]
        set story_drift [expr {abs($deltax/$deltay)}]

        # Compare the peak story drift to the previous largest and update if it is larger
        if {$story_drift > $max_drift} {
            set max_drift $story_drift
        }
    }

    return $max_drift
}
