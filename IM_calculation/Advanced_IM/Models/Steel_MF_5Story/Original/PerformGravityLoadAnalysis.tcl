#----------------------------------------------------------------------------------#
# PeformGravityLoadAnalysis.tcl
#	This module applies the previously defined gravity loads to the frame.  
#	This file should be executed before running the EQ or pushover.
#
# Units: kips, in, sec
#
# This file developed by: Curt Haselton of Stanford University
# Date: 10 June 2006
#
# Other files used in developing this model:
#	none
#----------------------------------------------------------------------------------#

#####################################################################################################
#####################################################################################################
### You can use this code to see how the building response when you apply gravity loads...
#####################################################################################################
## Did not work when I tried it on 6-12-06!
#
## Source in some commands to display the model
#	# a window to plot the nodal displacements versus load for node 36
#	recorder plot Node_$name.out Node3Xdisp 10 340 300 300 -columns 4 1 -columns 3 1 -columns 2 1	
#
#	# a window showing the displaced shape
#	recorder display g3 10 10 300 300 -wipe
#
#	# next three commmands define viewing system, all values in global coords
#	vrp 0.0 300.0 0    # point on the view plane in global coord, center of local viewing system
#	vup 0 1 0            # dirn defining up direction of view plane
#	vpn 0 0 1            # direction of outward normal to view plane
#
#	# next three commands define view, all values in local coord system
#	prp 0 0 100                   # eye location in local coord sys defined by viewing system
#	viewWindow -1600 1600 -600 600  # view bounds uMin, uMax, vMin, vMax in local coords
#	plane 0 150                   # distance to front and back clipping planes from eye
#	projection 0                  # projection mode
#
#	port -1 1 -1 1                # area of window that will be drawn into
#	fill 1                        # fill mode
#	display 1 0 10                
#
#####################################################################################################
#####################################################################################################


#############################################
# Apply gravity load pattern defined above

# Apply the loading
integrator LoadControl 0.20 1 0.20 0.20;	# Apply in 5 steps - I fixed this to apply the full
							#	amount of load on 2-19-05 (see hand notes if needed).

# Convergence test
#                  	tolerance maxIter displayCode
test RelativeNormDispIncr  1.0e-06     10         0

# Solution algorithm
algorithm Newton

# DOF numberer
numberer RCM

# Constraint handler
constraints Transformation

# System of equations solver
system SparseGeneral -piv

# Analysis for gravity load
analysis Static

initialize

# Perform the gravity load analysis
set ok [analyze 5];	# Apply in 5 steps, to be consistent with definition in integrator

# If it failed, stop the analysis and report an error
if {$ok != 0} {
	puts "*** Gravity Load application failed!!! "
	ERROR - stop analysis
}

loadConst -time 0.0;	# This sets all previous loads to be constant, so we don't incremet the gravity loads in future portions of the analysis
wipeAnalysis

# puts "Check: Gravity Load Applied!"

# Print element information - for testing
#print ele 1 2 3 4 5

#############################################


