#####################################
##  Vertical loading               ##
#####################################

# Reference lateral loads
pattern Plain 1 Constant {
#           eleID   uniform-                                           wy       wz    wx  (force/length)
       eleLoad	-ele	101	102	103	-type	-beamUniform	0	-6.78	0
       eleLoad	-ele	201	202	203	-type	-beamUniform	0	-9.07	0
       eleLoad	-ele	301	302	303	-type	-beamUniform	0	-10.22	0
       eleLoad	-ele	401	402	403	-type	-beamUniform	0	-15.23	0
       eleLoad	-ele	501	502	503	-type	-beamUniform	0	-6.78	0
       eleLoad	-ele	601	602	603	-type	-beamUniform	0	-9.38	0
       eleLoad	-ele	701	702	703	-type	-beamUniform	0	-17.88	0
       eleLoad	-ele	801	802	803	-type	-beamUniform	0	-11.01	0
       eleLoad	-ele	901 902	903	-type	-beamUniform	0	-15.12	0
       eleLoad	-ele	1001 1002 1003	-type	-beamUniform	0	-12.65	0
       eleLoad	-ele	1101 1102 1103	-type	-beamUniform	0	-8.32	0
       eleLoad	-ele	1201 1202 1203	-type	-beamUniform	0	-8.03	0
       eleLoad	-ele	1301 1302 1303	-type	-beamUniform	0	-9.38	0
       eleLoad	-ele	1401 1402 1403	-type	-beamUniform	0	-14.82	0
#              eleLoad	-ele	100	101	102	-type	-beamUniform	0	0 0
#              eleLoad	-ele	200	201	202	-type	-beamUniform	0	0 0
#              eleLoad	-ele	300	301	302	-type	-beamUniform	0	0 0
#              eleLoad	-ele	400	401	402	-type	-beamUniform	0	0 0
#              eleLoad	-ele	500	501	502	-type	-beamUniform	0	0 0
#              eleLoad	-ele	600	601	602	-type	-beamUniform	0	0 0
#              eleLoad	-ele	700	701	702	-type	-beamUniform	0	0 0
#              eleLoad	-ele	800	801	802	-type	-beamUniform	0	0 0
#              eleLoad	-ele	900	901	902	-type	-beamUniform	0	0 0
#              eleLoad	-ele	1000	1001	1002	-type	-beamUniform	0	0 0
#              eleLoad	-ele	1100	1101	1102	-type	-beamUniform	0	0 0
#              eleLoad	-ele	1200	1201	1202	-type	-beamUniform	0	0 0
#              eleLoad	-ele	1300	1301	1302	-type	-beamUniform	0	0 0
#              eleLoad	-ele	1400	1401	1402	-type	-beamUniform	0	0 0
}

 # added commands
# Recorders
set i 0	
foreach com $dir {	
        set com [lindex $dir $i]
	file mkdir $Output_path/$com/gravity_drift
	file mkdir $Output_path/$com/gravity_disp
	
	incr i
}	
#file mkdir $Output_path/gravity_dispNorm

# Recording gravity Disp/Drift in CM
# ---------------Displacement-----------------------
#  X dir
for {set story 1} {$story <= $num_stories} {incr story} {
    recorder Node -file $Output_path/000/gravity_disp/gr_dispCM_story${story}.out -time -node [lindex $ctrl_nodes $story ] -dof 1 disp
}

# Y dir
for {set story 1} {$story <= $num_stories} {incr story} {
    recorder Node -file $Output_path/090/gravity_disp/gr_dispCM_story${story}.out -time -node [lindex $ctrl_nodes $story ] -dof 2 disp
}
puts "----------gravity displacemant finished ----------"
# ----------------- Drift -------------------------  
# X dir
for {set story 1} {$story <= $num_stories} {incr story} {
   recorder Drift -file $Output_path/000/gravity_drift/gr_driftCM_story${story}.out -time -iNode [lindex $ctrl_nodes [expr {$story - 1}]] -jNode [lindex $ctrl_nodes $story] -dof 1 -perpDirn 3
}

# Y dir
for {set story 1} {$story <= $num_stories} {incr story} {
   recorder Drift -file $Output_path/090/gravity_drift/gr_driftCM_story${story}.out -time -iNode [lindex $ctrl_nodes [expr {$story - 1}]] -jNode [lindex $ctrl_nodes $story] -dof 2 -perpDirn 3
}
puts "---------gravity drift finished -----------------"
# ----------------- Norm Disp ----------------------
#for {set story 1} {$story <= $num_stories} {incr story} {
#    recorder Node -file $Output_path/gravity_dispNorm/gr_dispCM_N_story${story}.out -time -node [lindex $ctrl_nodes $story ] -dof 1 2 dispNorm
#}

# Recording gravity Disp/Drift in Corners
# ---------------------------------------------------------------------------------------------------------------------------------------
set i 1
foreach corner $ctrl_nodes_corner {
         
		 puts $corner
        #	
	# --------------- Drift Corner (1-4) ---------------------- 	 
	 # X dir
	for {set story 1} {$story <= $num_stories} {incr story} {
	   recorder Drift -file $Output_path/000/gravity_drift/gr_driftC${i}_story${story}.out -time -iNode [lindex [expr  $[lindex $corner 0]] [expr {$story - 1}]] -jNode [lindex [expr  $[lindex $corner 0]] $story] -dof 1 -perpDirn 3
	}    

	# Y dir
	for {set story 1} {$story <= $num_stories} {incr story} {
	   recorder Drift -file $Output_path/090/gravity_drift/gr_driftC${i}_story${story}.out -time -iNode [lindex [expr  $[lindex $corner 0]] [expr {$story - 1}]] -jNode [lindex [expr  $[lindex $corner 0]] $story] -dof 2 -perpDirn 3
	}     

	# --------------- Disp Corner (1-4)----------------------
	 # X dir
	for {set story 1} {$story <= $num_stories} {incr story} {
	   recorder Node -file $Output_path/000/gravity_disp/gr_dispC${i}_story${story}.out -time -node [lindex [expr  $[lindex $corner 0]] $story]  -dof 1 disp
	}

	# Y dir
	for {set story 1} {$story <= $num_stories} {incr story} {
	   recorder Node -file $Output_path/090/gravity_disp/gr_dispC${i}_story${story}.out -time -node [lindex [expr  $[lindex $corner 0]] $story]  -dof 2 disp
	}	   

	 # --------------- DispNorm Corner (1-4)----------------------
#	  for {set story 1} {$story <= $num_stories} {incr story} {
#	   recorder Node -file $Output_path/gravity_dispnorm/gr_dispC${i}_N_story${story}.out -time -node [lindex [expr  $[lindex $corner 0]] $story]  -dof 1 2 dispNorm
#	}	
incr i	
}	 
# ---------------------------------------------------------------------------------------------------------------------
   test NormDispIncr 1.0e-5 18 1
   #test EnergyIncr 1.0e-1 10 1
   algorithm Newton -initial 
   #algorithm Newton 
   system SparseGeneral -piv
   numberer RCM
   constraints Transformation #Lagrange
   integrator LoadControl 1e-4
   #integrator LoadControl 0.01
   analysis Static
   analyze 100
   setTime 0.0
   
   # added commands
   # loadConst -time 0.0;	# This sets all previous loads to be constant, so we don't incremet the gravity loads in future portions of the analysis
    # wipeAnalysis
    remove recorders
  # --------------------------------------------------------------------------------------------------------------------- 
  puts "Gravity is done!"
