# 
# define folders for recorders   
set i 0	
file mkdir $Output_path/rotd
file mkdir $Output_path/rotd/env_accl
file mkdir $Output_path/rotd/env_disp
file mkdir $Output_path/rotd/env_force 
# file mkdir $Output_path/rotd/disp
# file mkdir $Output_path/rotd/force  
# file mkdir $Output_path/rotd/accl	

#
# --------------------------------------------	
# --Define Time Series--
#

source [file join [file dirname [info script]] extra_rotd/time_series.tcl]


# puts "Output_path_resp = $Output_path_resp"
	
# Define Recorders
#
# --------------- Disp ----------------------
# time-series
# recorder Node -file $Output_path/rotd/disp/disp.out -time -node 2 -dof 1 disp


# Envelope
recorder EnvelopeNode -file $Output_path/rotd/env_disp/disp.out -time -node 2 -dof 1 disp

# --------------- Force ----------------------
# time-series
# recorder Node -file $Output_path/rotd/force/Force.out -time -node 1 -dof 1 reaction

# Envelope
recorder EnvelopeNode -file $Output_path/rotd/env_force/Force.out -time -node 1 -dof 1 reaction

# --------------- Acc ----------------------
# time-series
# recorder Node -file $Output_path/rotd/accl/accl.out -time -timeSeries 7 -node 2 -dof 1  accel 


# Envelope
recorder EnvelopeNode -file $Output_path/rotd/env_accl/accl.out -time -timeSeries 7 -node 2 -dof 1 accel	

# ----------------------------------------------
puts "Recorders are defined!"
# ----------------------------------------------


