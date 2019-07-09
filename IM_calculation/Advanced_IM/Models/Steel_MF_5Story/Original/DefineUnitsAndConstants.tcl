#----------------------------------------------------------------------------------#
# DefineUnitsAndConstants
#	This file is used to define units and constants.
#
# Units: kips, in, sec
#
# This file developed by: Curt Haselton of Stanford University
# Updated: 28 June 2005
# Date: March 10, 2004
#
# Other files used in developing this model:
#		none
#----------------------------------------------------------------------------------#

# Define units - With base units are kips, inches, and seconds
	set inch 1.0
	set kips 1.0
	set ft 12.0
	set lbs [expr 1/1000.0]
	#puts "Check lbs is $lbs "
	#puts "Check ft is $ft "
	set psf [expr ($lbs) * (1/$ft) * (1/$ft)]
	set pcf [expr ($lbs) * (1/$ft) * (1/$ft)* (1/$ft)]
	#puts "Check psf is $psf"
	set ksi 1.0
	set psi [expr 1/1000.0]
	set sec 1.0

# Define constants
	set pi 3.14159
	set hugeNumber 1000000000.0
	set g [expr 32.174 * $ft/($sec*$sec)]
#   puts $g

# puts "Units and constants defined"