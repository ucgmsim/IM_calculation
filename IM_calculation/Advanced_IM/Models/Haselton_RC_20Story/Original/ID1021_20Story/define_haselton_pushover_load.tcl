# Compute the first mode shape to be used as the lateral load pattern
    # Compute the sum of the displacements at all stories

    set modes [open  Analysis_Results/$modelname/Pushover/first_mode.txt w]
    set first_mode {}
    set first_mode_sum 0.0
    for {set story 1} {$story <= $num_stories} {incr story} {
        set story_disp [nodeEigenvector [lindex $ctrl_nodes $story] 1 1]
        lappend first_mode $story_disp
        set first_mode_sum [expr {$first_mode_sum + $story_disp}]
    }

puts $first_mode_sum
puts $num_stories

puts $modes $first_mode 
close $modes


    # Define the pushover load pattern such that the sum of all lateral loads equals 1. This allows
    # interpretation of the time step of the analysis as the applied load ratio.
    pattern Plain 2 Linear {
        for {set story 1} {$story <= $num_stories} {incr story} {
            for {set bay 1} {$bay <= 4} {incr bay} {
						
                load [expr {2020 +10*($story-1)}][expr (10*$bay+3)] \
                        [expr {[lindex $first_mode [expr {$story - 1}]]/$first_mode_sum/4.0}] 0.0 0.0
            }
        }
    }
