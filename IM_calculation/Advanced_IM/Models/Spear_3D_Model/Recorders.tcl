# 
# define folders for recorders
    #  file mkdir $Output_path 
    # file mkdir $Output_path/drift
set i 0	
foreach com $dir {	
        set com [lindex $dir $i]
	file mkdir $Output_path/$com/drift	
	file mkdir $Output_path/$com/accl	
	file mkdir $Output_path/$com/disp	
	file mkdir $Output_path/$com/env_drift
	file mkdir $Output_path/$com/env_disp
	file mkdir $Output_path/$com/env_accl 
	incr i
}
# commented out because of the changes of latest run.py	
#file mkdir $Output_path/env_dispNorm

# Recordes at Center of Mass
# --------------- Drift CM----------------------
# timeseries 
# X dir
for {set story 1} {$story <= $num_stories} {incr story} {
   recorder Drift -file $Output_path/000/drift/driftCM_story${story}.out -time -iNode [lindex $ctrl_nodes \
            [expr {$story - 1}]] -jNode [lindex $ctrl_nodes $story] -dof 1 -perpDirn 3
}    

# Y dir
for {set story 1} {$story <= $num_stories} {incr story} {
   recorder Drift -file $Output_path/090/drift/driftCM_story${story}.out -time -iNode [lindex $ctrl_nodes \
            [expr {$story - 1}]] -jNode [lindex $ctrl_nodes $story] -dof 2 -perpDirn 3
} 

# Peak values
# X dir
for {set story 1} {$story <= $num_stories} {incr story} {
   recorder EnvelopeDrift -file $Output_path/000/env_drift/driftCM_story${story}.out -time -iNode [lindex $ctrl_nodes \
            [expr {$story - 1}]] -jNode [lindex $ctrl_nodes $story] -dof 1 -perpDirn 3
}    

# Y dir
for {set story 1} {$story <= $num_stories} {incr story} {
   recorder EnvelopeDrift -file $Output_path/090/env_drift/driftCM_story${story}.out -time -iNode [lindex $ctrl_nodes \
            [expr {$story - 1}]] -jNode [lindex $ctrl_nodes $story] -dof 2 -perpDirn 3
} 

# --------------- Acc CM----------------------
# timeseries 
# X dir
for {set story 0} {$story <= $num_stories} {incr story} {
    recorder Node -file $Output_path/000/accl/acclCM_story${story}.out -time -timeSeries 5 -node [lindex $ctrl_nodes $story] -dof 1  accel 
}

# Y dir
for {set story 0} {$story <= $num_stories} {incr story} {
    recorder Node -file $Output_path/090/accl/acclCM_story${story}.out -time -timeSeries 6 -node [lindex $ctrl_nodes $story] -dof 2  accel 
}

# Peak values
# X dir
for {set story 0} {$story <= $num_stories} {incr story} {
   recorder EnvelopeNode -file $Output_path/000/env_accl/acclCM_story${story}.out -time -timeSeries 5 -node [lindex $ctrl_nodes \
            $story]  -dof 1 accel 
}

# Y dir
for {set story 0} {$story <= $num_stories} {incr story} {
   recorder EnvelopeNode -file $Output_path/090/env_accl/acclCM_story${story}.out -time -timeSeries 6 -node [lindex $ctrl_nodes \
            $story]  -dof 2 accel 
}

# --------------- Disp CM----------------------
# timeseries 
# X dir
for {set story 0} {$story <= $num_stories} {incr story} {
    recorder Node -file $Output_path/000/disp/dispCM_story${story}.out -time  -node [lindex $ctrl_nodes $story] -dof 1  disp 
}

# Y dir
for {set story 0} {$story <= $num_stories} {incr story} {
    recorder Node -file $Output_path/090/disp/dispCM_story${story}.out -time  -node [lindex $ctrl_nodes $story] -dof 2  disp  
}

# Peak values
# X dir
for {set story 1} {$story <= $num_stories} {incr story} {
   recorder EnvelopeNode -file $Output_path/000/env_disp/dispCM_story${story}.out -time -node [lindex $ctrl_nodes \
            $story]  -dof 1 disp
}

# Y dir
for {set story 1} {$story <= $num_stories} {incr story} {
   recorder EnvelopeNode -file $Output_path/090/env_disp/dispCM_story${story}.out -time -node [lindex $ctrl_nodes \
            $story]  -dof 2 disp
}	

# --------------- DispNorm CM----------------------
# commented out because the lastest version calculate this differently in run.py
#  for {set story 1} {$story <= $num_stories} {incr story} {
#   recorder EnvelopeNode -file $Output_path/env_dispNorm/dispCM_N_story${story}.out -time -node [lindex $ctrl_nodes \
#            $story]  -dof 1 2 dispNorm
#}	

# Recordes at Corners  	
# --------------- Corners 1,2,3,4 (nodes 1,5,10,12) ----------------------
	
set i 1
set st 1
foreach corner $ctrl_nodes_corner {
         
		 puts $corner
        	
	# --------------- Drift Corner (1-4) ---------------------- 	
	  # timeseries
           # X dir
	for {set story 1} {$story <= $num_stories} {incr story} {
	   recorder Drift -file $Output_path/000/drift/driftC${i}_story${story}.out -time -iNode [lindex [expr  $[lindex $corner 0]]\
				[expr {$story - 1}]] -jNode [lindex [expr  $[lindex $corner 0]] $story] -dof 1 -perpDirn 3
	}    

	# Y dir
	for {set story 1} {$story <= $num_stories} {incr story} {
	   recorder Drift -file $Output_path/090/drift/driftC${i}_story${story}.out -time -iNode [lindex [expr  $[lindex $corner 0]] \
				[expr {$story - 1}]] -jNode [lindex [expr  $[lindex $corner 0]] $story] -dof 2 -perpDirn 3
	}     
	
	 # Peak values
	 # X dir
	for {set story 1} {$story <= $num_stories} {incr story} {
	   recorder EnvelopeDrift -file $Output_path/000/env_drift/driftC${i}_story${story}.out -time -iNode [lindex [expr  $[lindex $corner 0]]\
				[expr {$story - 1}]] -jNode [lindex [expr  $[lindex $corner 0]] $story] -dof 1 -perpDirn 3
	}    

	# Y dir
	for {set story 1} {$story <= $num_stories} {incr story} {
	   recorder EnvelopeDrift -file $Output_path/090/env_drift/driftC${i}_story${story}.out -time -iNode [lindex [expr  $[lindex $corner 0]] \
				[expr {$story - 1}]] -jNode [lindex [expr  $[lindex $corner 0]] $story] -dof 2 -perpDirn 3
	}     

	# --------------- Acc Corner (1-4)----------------------
         # timeseries
	  # X dir
	for {set story 0} {$story <= $num_stories} {incr story} {
	   recorder Node -file $Output_path/000/accl/acclC${i}_story${story}.out -time -timeSeries 5 -node [lindex [expr  $[lindex $corner 0]] \
				$story]  -dof 1 accel 
	}

	# # Y dir
	for {set story 0} {$story <= $num_stories} {incr story} {
	   recorder Node -file $Output_path/090/accl/acclC${i}_story${story}.out -time -timeSeries 6 -node [lindex [expr  $[lindex $corner 0]] \
				$story]  -dof 2 accel 
	}
	
       # Peak values
	   # X dir
	for {set story 0} {$story <= $num_stories} {incr story} {
	   recorder EnvelopeNode -file $Output_path/000/env_accl/acclC${i}_story${story}.out -time -timeSeries 5 -node [lindex [expr  $[lindex $corner 0]] \
				$story]  -dof 1 accel 
	}

	  # Y dir
	for {set story 0} {$story <= $num_stories} {incr story} {
	   recorder EnvelopeNode -file $Output_path/090/env_accl/acclC${i}_story${story}.out -time -timeSeries 6 -node [lindex [expr  $[lindex $corner 0]] \
				$story]  -dof 2 accel 
	}

	# --------------- Disp Corner (1-4)----------------------
	# timeseries
	# X dir
	for {set story 1} {$story <= $num_stories} {incr story} {
	   recorder Node -file $Output_path/000/disp/dispC${i}_story${story}.out -time -node [lindex [expr  $[lindex $corner 0]]\
				$story]  -dof 1 disp
	}

	# Y dir
	for {set story 1} {$story <= $num_stories} {incr story} {
	   recorder Node -file $Output_path/090/disp/dispC${i}_story${story}.out -time -node [lindex [expr  $[lindex $corner 0]] \
				$story]  -dof 2 disp
	}	  
			
	# Peak values
	 # X dir
	for {set story 1} {$story <= $num_stories} {incr story} {
	   recorder EnvelopeNode -file $Output_path/000/env_disp/dispC${i}_story${story}.out -time -node [lindex [expr  $[lindex $corner 0]]\
				$story]  -dof 1 disp
	}

	# Y dir
	for {set story 1} {$story <= $num_stories} {incr story} {
	   recorder EnvelopeNode -file $Output_path/090/env_disp/dispC${i}_story${story}.out -time -node [lindex [expr  $[lindex $corner 0]] \
				$story]  -dof 2 disp
	}	   

	 # --------------- DispNorm Corner (1-4)----------------------
# commented out because the lastest version calculate differently in run.py
#	  for {set story 1} {$story <= $num_stories} {incr story} {
#	   recorder EnvelopeNode -file $Output_path/env_dispNorm/dispC${i}_N_story${story}.out -time -node [lindex [expr  $[lindex $corner 0]] \
#				$story]  -dof 1 2 dispNorm
#	}	
	
incr i	
}	 

  
  # ----------------------------------------------
puts "Recorders are defined!"
# ----------------------------------------------
