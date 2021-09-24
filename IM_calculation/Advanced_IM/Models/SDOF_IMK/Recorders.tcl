# 
# define folders for recorders   
	file mkdir $Output_path/env_accl
	file mkdir $Output_path/env_disp
	file mkdir $Output_path/env_force 
	# file mkdir $Output_path/disp
	# file mkdir $Output_path/force  
    # file mkdir $Output_path/accl	
#
# --------------------------------------------	
# --Define Time Series--
#

source [file join [file dirname [info script]] ../general/time_series.tcl]
	
# Define Recorders
#
# --------------- Disp ----------------------
# time-series
# recorder Node -file $Output_path/disp/disp.out -time -node 2 -dof 1 disp


# Envelope
recorder EnvelopeNode -file $Output_path/env_disp/disp.out -time -node 2 -dof 1 disp

# --------------- Force ----------------------
# time-series
# recorder Node -file $Output_path/force/Force.out -time -node 1 -dof 1 reaction

# Envelope
recorder EnvelopeNode -file $Output_path/env_force/Force.out -time -node 1 -dof 1 reaction

# --------------- Acc ----------------------
# time-series
# recorder Node -file $Output_path/accl/accl.out -time -timeSeries 5 -node 2 -dof 1  accel 


# Envelope
recorder EnvelopeNode -file $Output_path/env_accl/accl.out -time -timeSeries 5 -node 2 -dof 1 accel	

# ----------------------------------------------
puts "Recorders are defined!"
# ----------------------------------------------


