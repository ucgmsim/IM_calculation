#----------------------------------------------------------------------------------#
# DefineFunctionsAndProcedures
#		In this module, functions and procedures are defined, that are used in
#			the rest of the program.
#		Buffalo: I did document this file a bit, but not as well as other files.
#			If you need more explination of what I am doing, let me know and I 
#			will document the file better.
#
# Units: kips, in, sec
#
# This file developed by: Curt Haselton of Stanford University
# Updated: 28 June 2005
# Date: 17 Sept 2003
#
# Other files used in developing this model:
#		test_simPlanned.tcl by Paul Cordova of Stanford University
#----------------------------------------------------------------------------------#

proc RunEQLoading {eqNumber saTOneForRun dtForAnalysis alpha1 alpha2 g testType allowNormDisplIncrForConv analysisType EQsolutionAlgorithm iterAlgo iterAlgoArg eqRecordListToUse constraintForEQ constraintArg1EQ constraintArg2EQ nodeNumsAtEachFloorLIST minStoryDriftRatioForCollapse floorHeightsLIST listOfElementsForRayleighDamping } {

# DISCLAMER! - 
#	How I currently have this solution algorithm set up (it allows the convergence tolerance to increase a lot), 
#	it will almost always allow convergence, so it is the responsibility of the user to check the output and 
#	see the maximum toalerances (in the output files) that were used and decide if this is ok.  This usually is
#	alright, but should be checked to be sure.


# eqFileName - specifies name of the record (file name) to be run - this file must be in the EQs folder
# alpha1, alpha2 - the damping coefficients
# scaleFactor - the scaling factor on the record (ex. 1, 1.5, 2)
# dT - the time step used for integration
# numPointsForEQ - the number of points for the eq record
# g - gravitational constant
# testType - convergence test to be used - e.g. normDisplIncr
# allowNormDisplIncrForConv - tells whther or not NormDisplIncr is allowed if convergence is bad - used below in solution algorithm
# analysisType - this is just used for a folder name for outputing whether NormDisplIncr was used or not.
# EQsolutionAlgorithm - the solution algorithm to use
# displayMode - if the display is on or off

#################################
# Make the directory for RunInformation, to output convergence information into.  Also open a file stream for the convergence log
		set startDir [pwd]
		cd ..
		cd ..
			
		set baseDir [pwd]
		cd $baseDir/Output/$analysisType/EQ_$eqNumber/Sa_$saTOneForRun
		set runDir [pwd]

		file mkdir RunInformation; 
		cd $runDir/RunInformation/

		set convLogFilename 	[open convLogFileOUT.out w]
		set convPlotFilename	[open convPlotFileOUT.out w]

		cd $startDir 
##################################

#puts "EQ number is: $eqNumber"
puts "Current directory is: [pwd]"

# SOurce in the information for the EQ's
source DefineEarthquakeRecordInformation.tcl

# Compute scale factor - if it's a single component, scale it by the Sa of that component, if it's geoMean, then 
#	scale it by the geometric mean of the two horizontal components.

	# Even though this is not the collapse analysis, I copied a variable from the collapse analysis script, so just 
	#	rename my variable here so it will work!
	set eqFormatForCollapseList $eqRecordListToUse


if {$eqRecordListToUse == "Formatted_singleComp" || $eqRecordListToUse == "PEER_singleComp" || $eqRecordListToUse == "PEER-NGA_singleComp" || $eqRecordListToUse == "PEER-NGA_Rotated_singleComp"} { 
	set scaleFactor [expr $saTOneForRun / $saTOneForEQ($eqNumber)]

} elseif {$eqRecordListToUse == "Formatted_geoMean" || $eqRecordListToUse == "PEER_geoMean" || $eqRecordListToUse == "PEER-NGA_geoMean" || $eqRecordListToUse == "PEER-NGA_Rotated_geoMean"} {
	# The Eq number that we are running is for a single component, so we need to 
	# 	compute the number that I used that is generic to both component (the number that is about 10x less)
	# 	and get the GeoMean with that number.
	set eqNumberForGeoMean [expr $eqNumber / 10]
	set scaleFactor [expr $saTOneForRun / $saTOneForEQGeoMean($eqNumberForGeoMean)]
	puts "eqNumberForGeoMean is $eqNumberForGeoMean"
	puts "scaleFactor is $scaleFactor"

} elseif {$eqRecordListToUse == "Formatted_codeScaling" || $eqRecordListToUse == "PEER_codeScaling" || $eqRecordListToUse == "PEER-NGA_codeScaling" || $eqRecordListToUse == "PEER-NGA_Rotated_codeScaling"} {
	# The Eq number that we are running is for a single component, so we need to 
	# 	compute the number that I used that is generic to both component (the number that is about 10x less)
	# 	and get the GeoMean with that number.
	set eqNumberForCodeScaling [expr $eqNumber / 10]
	set scaleFactor [expr ($saTOneForRun * $scaleFactorForMeanOfCodeGMSetToBeOne($eqNumberForCodeScaling))]
	puts "eqNumberForCodeScaling is $eqNumberForCodeScaling";
	puts "scaleFactor is $scaleFactor"

} else {
	puts "ERROR - eqRecordListToUse not found!"
	ERROR
}

puts "Running EQ for: EQ $eqNumber at Sa of $saTOneForRun, with scale factor $scaleFactor"

# Initialize the value to show if it had to use the NormDisplIncr or not
set usedNormDisplIncr 0
set isSingularR 0;			# Initialize
set isCollapsedForCurrentRun 0;	# Initialize

	# Depending on what type of record format is being used, then do different things to set up the record to run...
	if {$eqRecordListToUse == "Formatted_singleComp" || $eqRecordListToUse == "Formatted_geoMean" || $eqRecordListToUse == "Formatted_codeScaling"} {
		# Run formatted record
		set Series "Path -filePath C:/OpenSeesProcessingFiles/EQs/$eqFileName($eqNumber) -dt $dtForEQ($eqNumber) -factor [expr $scaleFactor * $g]"
		set numPoints 	[expr $numPointsForEQ($eqNumber)]
		set dt		[expr $dtForEQ($eqNumber)]
	} elseif {$eqRecordListToUse == "PEER_singleComp" || $eqRecordListToUse == "PEER_geoMean" || $eqRecordListToUse == "PEER-NGA_singleComp" || $eqRecordListToUse == "PEER-NGA_geoMean" || $eqRecordListToUse == "PEER-NGA_Rotated_singleComp" || $eqRecordListToUse == "PEER-NGA_Rotated_geoMean" || $eqRecordListToUse == "PEER_codeScaling" || $eqRecordListToUse == "PEER-NGA_codeScaling" || $eqRecordListToUse == "PEER-NGA_Rotated_codeScaling"} {
		# Run PEER formatted record
		###########
		# Use the procedure to get dt and numPoints for the EQ and make a file for the EQ that has a single column of accel TH in it.
			# Initialize the dt and numPoints for the current EQ.  These are passed-by-reference to the procedure, so the proc
			#	changes thier values.
			set dt 		0.0;	# This will be changed by ReadSMDFile2 (pass-by-reference)
			set numPoints	0;	# This will be changed by ReadSMDFile2 (pass-by-reference)
			set eqFileNameForProc "C:/OpenSeesProcessingFiles/EQs/$eqFileName($eqNumber)"
			set outputFileNameForProc "C:/OpenSeesProcessingFiles/EQs/SortedEQFile_($eqNumber).txt"
		
			
			# Call the procedure to do the "pre-processing"; depends on file format
			if { $eqRecordListToUse == "PEER_singleComp" || $eqRecordListToUse == "PEER_geoMean" || $eqRecordListToUse == "PEER_codeScaling" } {
				# Call the file that processes the records in the original PEER format
				ReadSMDFile2 $eqFileNameForProc $outputFileNameForProc $dt $numPoints
			} elseif { $eqRecordListToUse == "PEER-NGA_singleComp" || $eqRecordListToUse == "PEER-NGA_geoMean" || $eqRecordListToUse == "PEER-NGA_codeScaling"} {
				# Call the file that processes the records in the new PEER-NGA format
				ReadSMDFile_PEER-NGA-Format $eqFileNameForProc $outputFileNameForProc $dt $numPoints
			} elseif { $eqRecordListToUse == "PEER-NGA_Rotated_singleComp" || $eqRecordListToUse == "PEER-NGA_Rotated_geoMean" || $eqRecordListToUse == "PEER-NGA_Rotated_codeScaling"} {
				# Call the file that processes the records in the new PEER-NGA format
				ReadSMDFile_PEER-NGA-Rotated-Format $eqFileNameForProc $outputFileNameForProc $dt $numPoints
			} else {
				puts "ERROR - file format input not valid!"
				ERROR - stop analysis
			}

#			puts "dt is $dt"
#			puts "numPoints is $numPoints"

		###########

		set Series "Path -filePath C:/OpenSeesProcessingFiles/EQs/SortedEQFile_($eqNumber).txt -dt $dt -factor [expr $scaleFactor * $g]"
	} else {
		puts "ERROR: The EQ record format is not defined correctly!"
		ERROR - stop analysis
	}

set tag 101

#                           tag  dir  accel    series
pattern UniformExcitation  $tag   1  -accel   $Series

set ok 0
set currentTime [getTime]
set maxTime [expr [getTime]+[expr ($numPoints)*$dt]];  
set numSteps [expr 1 * [expr ($numPoints)*$dt] / [expr ($dtForAnalysis)]]

# Source in the solution algorithm to carry out the solution
source SolutionAlgorithm.tcl



# Output file saying whether or not it used NormDisplIncr
			set startDir [pwd]
			cd ..
			cd ..
			
			set baseDir [pwd]
			cd $baseDir/Output/$analysisType/EQ_$eqNumber/Sa_$saTOneForRun
			set runDir [pwd]

			file mkdir RunInformation; 
			cd $runDir/RunInformation/

			set filenameEQ [open usedNormDisplIncrOUT.out w]
			puts $filenameEQ $usedNormDisplIncr
			close $filenameEQ 

			# Output the scaled Sa,comp
			set saCompScaled [expr $saTOneForEQ($eqNumber) * $scaleFactor];
			set filename [open saCompScaledOUT.out w]
			puts $filename $saCompScaled
			close $filename 

			# Output the scaled Sa,geoMean only if we scaled by this (b/c if we used Sa,comp, we may not have define the geoMean information)
			if {$eqRecordListToUse == "Formatted_geoMean" || $eqRecordListToUse == "PEER_geoMean" || $eqRecordListToUse == "PEER-NGA_geoMean" || $eqRecordListToUse == "PEER-NGA_Rotated_geoMean"} {			
				set eqNumberForGeoMean [expr $eqNumber / 10]
				set saGeoMeanScaled [expr $saTOneForEQGeoMean($eqNumberForGeoMean) * $scaleFactor];
				set filename [open saGeoMeanScaledOUT.out w]
				puts $filename $saGeoMeanScaled
				close $filename 
                	} else {
                      	puts "Not putting geoMean is output file"
				set filename [open saGeoMeanScaledOUT.out w]
				puts $filename -1
				close $filename 
			}



			# Output the scale factor
			set filename [open scaleFactorAppliedToCompOUT.out w]
			puts $filename $scaleFactor
			close $filename 

			# Output the file format and scaling mthod used
			set filename [open eqRecordListToUseOUT.out w]
			puts $filename $eqRecordListToUse 
			close $filename 


			cd $startDir 
}


#####################################################################################################
#####################################################################################################
####################################################################################################
####################################################################################################

proc RunEQLoadingForCollapse {eqNumber saTOneForRun scaleFactorForRunFromMatlab saCompScaled saGeoMeanScaled dtForAnalysis alpha1 alpha2 g testType allowNormDisplIncrForConv analysisType EQsolutionAlgorithm iterAlgo iterAlgoArg nodeNumsAtEachFloorLIST minStoryDriftRatioForCollapse useINDAsCollapse eqFormatForCollapseList floorHeightsLIST isCollapsed isSingular isNonConv maxStoryDriftRatioForFullStr minStoryDriftRatioForFullStr constraintForEQ constraintArg1EQ constraintArg2EQ listOfElementsForRayleighDamping } {


# This is just like RunEQLoading, except the algorithm stops if collapse is detected (based on input into SetAnalysisOptions).
#	This algorithm is currently specific to ONLY A FOUR STORY STRUCTURE (due to the checking for collapse and singularity
#	at each step of the analysis)!

# eqFileName - specifies name of the record (file name) to be run - this file must be in the EQs folder
# alpha1, alpha2 - the damping coefficients
# scaleFactor - the scaling factor on the record (ex. 1, 1.5, 2)
# dT - the time step used for integration
# numPointsForEQ - the number of points for the eq record
# g - gravitational constant
# testType - convergence test to be used - e.g. normDisplIncr
# allowNormDisplIncrForConv - tells whther or not NormDisplIncr is allowed if convergence is bad - used below in solution algorithm
# analysisType - this is just used for a folder name for outputing whether NormDisplIncr was used or not.
# EQsolutionAlgorithm - the solution algorithm to use
# displayMode - if the display is on or off
###################################################################################################

# First define the variables that will be passed back to the calling function using upvar...
	# Buffalo - you don't need to knoww hat this is doing.
	# These variables are used to try to find in the building is collapse and/or singular at each step of the analysis.
	# The "R" simply means that it is a returned value
	upvar isCollapsed isCollapsedR;
	upvar isSingular isSingularR;
	upvar isNonConv isNonConvR;
	upvar maxStoryDriftRatioForFullStr maxStoryDriftRatioForFullStrR;
	upvar minStoryDriftRatioForFullStr minStoryDriftRatioForFullStrR;

#################################
# Make the directory for RunInformation, to output convergence information into.  Also open a file stream for the convergence log
		set startDir [pwd]
		cd ..
		cd ..
			
		set baseDir [pwd]
		cd $baseDir/Output/$analysisType/EQ_$eqNumber/Sa_$saTOneForRun
		set runDir [pwd]

		file mkdir RunInformation; 
		cd $runDir/RunInformation/

		set convLogFilename 	[open convLogFileOUT.out w]
		set convPlotFilename	[open convPlotFileOUT.out w]

		cd $startDir 
##################################

#puts "EQ number is: $eqNumber"
#puts "Current directory is: [pwd]"

# Source in the information for the EQ's
source DefineEarthquakeRecordInformation.tcl

## The scale factor is now computed in Matlab and sent to Opensees and this function; therefore, we do nto need to compute it anymore!! (6-29-06)
#if {$eqFormatForCollapseList == "Formatted_singleComp" || $eqFormatForCollapseList == "PEER_singleComp" || $eqFormatForCollapseList == "PEER-NGA_singleComp" || $eqFormatForCollapseList == "PEER-NGA_Rotated_singleComp"} { 
#	set scaleFactor [expr $saTOneForRun / $saTOneForEQ($eqNumber)]
#} elseif {$eqFormatForCollapseList == "Formatted_geoMean" || $eqFormatForCollapseList == "PEER_geoMean" || $eqFormatForCollapseList == "PEER-NGA_geoMean" || $eqFormatForCollapseList == "PEER-NGA_Rotated_geoMean"} {
#	# The Eq number that we are running is for a single component, so we need to 
#	# 	compute the number that I used that is generic to both component (the number that is about 10x less)
#	# 	and get the GeoMean with that number.
#	set eqNumberForGeoMean [expr $eqNumber / 10]
#	set scaleFactor [expr $saTOneForRun / $saTOneForEQGeoMean($eqNumberForGeoMean)]
#	puts "eqNumberForGeoMean is $eqNumberForGeoMean"
#	puts "scaleFactor is $scaleFactor"
#
#} elseif {$eqFormatForCollapseList == "Formatted_codeScaling" || $eqFormatForCollapseList == "PEER_codeScaling" || $eqFormatForCollapseList == "PEER-NGA_codeScaling" || $eqFormatForCollapseList == "PEER-NGA_Rotated_codeScaling"} {
#	# The Eq number that we are running is for a single component, so we need to 
#	# 	compute the number that I used that is generic to both component (the number that is about 10x less)
#	# 	and get the GeoMean with that number.
#	set eqNumberForCodeScaling [expr $eqNumber / 10]
#	set scaleFactor [expr ($saTOneForRun * $scaleFactorForMeanOfCodeGMSetToBeOne($eqNumberForCodeScaling))]
#	puts "eqNumberForCodeScaling is $eqNumberForCodeScaling";
#	puts "scaleFactor is $scaleFactor"
#
#} else {
#	puts "ERROR - eqRecordListToUse not found!"
#	ERROR
#}
#

# Output some stuff
	puts "Running EQ for: EQ $eqNumber at Sa of $saTOneForRun, with scale factor $scaleFactorForRunFromMatlab"
	puts "saTOneForRun is $saTOneForRun"
	puts "scaleFactorForRunFromMatlab is $scaleFactorForRunFromMatlab"
	puts "saCompScaled is $saCompScaled"
	puts "saGeoMeanScaled is $saGeoMeanScaled"

# Initialize variables used for the analysis
	set usedNormDisplIncr 			0;			# Did not yet need to use NormDisplIncr to converge
	set isCollapsedR 				0;			# Not yet collapsed
	set isSingularR 				0;			# Not yet singular
	set isNonConvR 				0;			# Not yet non-converged
	set maxStoryDriftRatioForFullStrR 	0;			# No drift yet
	set minStoryDriftRatioForFullStrR 	0;			# No drift yet
	set isCollapsedForCurrentRun 0;				# Initialize - I am not sure if I need this or not

# Depending on what type of record format is being used, then do different things to set up the record to run...
	if {$eqFormatForCollapseList == "Formatted_singleComp" || $eqFormatForCollapseList == "Formatted_geoMean" || $eqFormatForCollapseList == "Formatted_codeScaling"} {
		# Run formatted record
		set Series "Path -filePath C:/OpenSeesProcessingFiles/EQs/$eqFileName($eqNumber) -dt $dtForEQ($eqNumber) -factor [expr $scaleFactorForRunFromMatlab * $g]"
		set numPoints 	[expr $numPointsForEQ($eqNumber)]
		set dt		[expr $dtForEQ($eqNumber)]
	} elseif {$eqFormatForCollapseList == "PEER_singleComp" || $eqFormatForCollapseList == "PEER_geoMean" || $eqFormatForCollapseList == "PEER-NGA_singleComp" || $eqFormatForCollapseList == "PEER-NGA_geoMean" || $eqFormatForCollapseList == "PEER-NGA_Rotated_singleComp" || $eqFormatForCollapseList == "PEER-NGA_Rotated_geoMean" || $eqFormatForCollapseList == "PEER_codeScaling" || $eqFormatForCollapseList == "PEER-NGA_codeScaling" || $eqFormatForCollapseList == "PEER-NGA_Rotated_codeScaling"} {
		# Run PEER formatted record
		###########
		# Use the procedure to get dt and numPoints for the EQ and make a file for the EQ that has a single column of accel TH in it.
			# Initialize the dt and numPoints for the current EQ.  These are passed-by-reference to the procedure, so the proc
			#	changes thier values.
			set dt 		0.0;	# This will be changed by ReadSMDFile2 (pass-by-reference)
			set numPoints	0;	# This will be changed by ReadSMDFile2 (pass-by-reference)
			set eqFileNameForProc "C:/OpenSeesProcessingFiles/EQs/$eqFileName($eqNumber)"
			set outputFileNameForProc "C:/OpenSeesProcessingFiles/EQs/SortedEQFile_($eqNumber).txt"
		
			
			# Call the procedure to do the "pre-processing"; depends on file format
			if { $eqFormatForCollapseList == "PEER_singleComp" || $eqFormatForCollapseList == "PEER_geoMean" || $eqFormatForCollapseList == "PEER_codeScaling" } {
				# Call the file that processes the records in the original PEER format
				ReadSMDFile2 $eqFileNameForProc $outputFileNameForProc $dt $numPoints
			} elseif { $eqFormatForCollapseList == "PEER-NGA_singleComp" || $eqFormatForCollapseList == "PEER-NGA_geoMean" || $eqFormatForCollapseList == "PEER-NGA_codeScaling"} {
				# Call the file that processes the records in the new PEER-NGA format
				ReadSMDFile_PEER-NGA-Format $eqFileNameForProc $outputFileNameForProc $dt $numPoints
			} elseif { $eqFormatForCollapseList == "PEER-NGA_Rotated_singleComp" || $eqFormatForCollapseList == "PEER-NGA_Rotated_geoMean" || $eqFormatForCollapseList == "PEER-NGA_Rotated_codeScaling"} {
				# Call the file that processes the records in the new PEER-NGA format
				ReadSMDFile_PEER-NGA-Rotated-Format $eqFileNameForProc $outputFileNameForProc $dt $numPoints
			} else {
				puts "ERROR - file format input not valid!"
				ERROR - stop analysis
			}

#			puts "dt is $dt"
#			puts "numPoints is $numPoints"

		###########

		set Series "Path -filePath C:/OpenSeesProcessingFiles/EQs/SortedEQFile_($eqNumber).txt -dt $dt -factor [expr $scaleFactorForRunFromMatlab * $g]"
	} else {
		puts "ERROR: The EQ record format is not defined correctly!"
		ERROR - stop analysis
	}


# Set the tab number for the EQ pattern
	set tag 101

# Define the loading patern for the EQ
	#                           tag  dir  accel    series
	pattern UniformExcitation  $tag   1  -accel   $Series

# Initial variables before starting dynamic analysis
	set ok 0
	set currentTime [getTime]
	set maxTime [expr [getTime]+[expr ($numPoints)*$dt]];  
	set numSteps [expr 1 * [expr ($numPoints)*$dt] / [expr ($dtForAnalysis)]]

# Define the ranges of tolerances to try for convergence.  Note that the maximum tolerance 
#	used in the analsyis will be output in the output files.
	set testTolerance 1.0e-6
	set testMinTolerance1 1.0e-5;	
	set testMinTolerance2 1.0e-4;	
	set testMinTolerance3 1.0e-3;	
	set testMinTolerance4 1.0e-2;	
	set testMinTolerance5 1.0e-1;	
		
# Define the iteration information used for difference situations and tests
	set testIterations 100;			# Changed on 1-7-05 for Corotational transformation
	set testInitialIterations 1000
	set testLowIter 10;	# Used to try each test in the loop
	set ratioForInitialAlgo	200;	# This is the ratio of testIterations that is allowed for -initial test 
	set testHighIter 1000;	# Used to try to make it converge at the very end
	
# Define the initial test, this will be changed later during convergance work.
	test $testType $testTolerance $testIterations 0

# Set initialize the maximum tolerance used - for output to know about what tolerance was used to obtain convergence.
	set maxTolUsed $testTolerance;


##########################################################################
# Decide which solution algoritm to use - based on the analysis options

if {$EQsolutionAlgorithm == "CordovaFrank"} {
	puts "EQ Solution Algorithm: CordovaFrank"

	# Use this alogorithm!  This is the best one!

	# This is the same as CordovaMod, except that I use Franks adaptative proceedure when calling the "analyze" command.

	# I changed this to Line Search (it was just Newton) to see if it will make it work (on 6-30-04)
	#     Using the Newton for the first step made the model run for Paul, but it
	#	made my model (when I added the gravity frames) not work at all).  Now 
	#	that I am using NewtonLineSearch from the start, then it's working!!!
	#algorithm Newton
	#algorithm NewtonLineSearch 0.6
	#algorithm NewtonLineSearch 0.8
	#algorithm Newton -initial
	#algorithm KrylovNewton 
	#algorithm KrylovNewton -initial


	##########
	# Set options to use Franks adaptative analysis
		#analyze $numIncr <$dt> <$dtMin $dtMax $Jd>
	set numIncrForFranksAnalysis	1;	# This is the number of increments that are used before I enter my convergence loop 
							# BE SURE TO DO 1 so it will pop out and I can check the displ. for possible 
							# 	collapse and for singularity AT EVERY STEP (changed on 1-21-05).
	set dtFrank				[expr $dtForAnalysis];
	set dtMinFrank			[expr $dtForAnalysis / 100.0];
	set dtMaxFrank			[expr $dtForAnalysis];
	##########

	system UmfPack

	# Define integrator
	integrator Newmark 0.5  0.25 

	# Define damping - this is done with VB now
	#source DefineDampingObjects.tcl

	# DOF numberer
	numberer RCM

	# Constraint handler
#	constraints Transformation
	constraints $constraintForEQ $constraintArg1EQ $constraintArg2EQ
#	constraints Penalty 1.0e15 1.0e15

	# Create the analysis object
	analysis Transient

	# I took this out, so all of the steps are in the analysis loop
	#analyze 1 $dtForAnalysis;	# Paul took one step with Newton when he did his model.

	# switch to NewtonLineSearch - [[ Paul's old comment - "for some reason this does not work 
	# on the very first step, but fine for all succeeding steps"]].

	# Changed here with Frank on 7-6-04 (he reocmended using Newton instead of LineSearch)
	algorithm $iterAlgo

	#Analysis loop - this wil complete when the EQ is over, the structure collapses, or the system goes singular
	while {$ok == 0 && $currentTime < $maxTime && $isCollapsedR == 0 && $isSingularR == 0} {

      	puts "tFinal is $maxTime; and tCurrent is $currentTime"

		# Do step with initial tolerance and input algorithm
		test $testType $testTolerance $testIterations 0


		# Altered here for Cordova Frank
#    		set ok [analyze 1 $dtForAnalysis]
		set ok [analyze $numIncrForFranksAnalysis $dtFrank $dtMinFrank $dtMaxFrank]


		# Keep track of the maximum tolarance used in this step  - for the convPlotFile.  This is later increased it necessary.
		set maxTolUsedInCurrentStep 	[expr $testTolerance]

		#### Change things for convergence ###############
		# If it's not ok, try to decrease dT, but keep the toerance the samecall another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testTolerance]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/10];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, try to decrease dT a bit more, but keep the toerance the samecall another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testTolerance]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/20];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, try to decrease dT a bit more, but keep the toerance the samecall another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testTolerance]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/40];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, try to decrease dT a bit more, but keep the toerance the samecall another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testTolerance]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/80];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance1...call another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testMinTolerance1]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/10];	# This was /20, so maybe change back?
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance1...call another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testMinTolerance1]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/20];	# This was /20, so maybe change back?
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance1...call another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testMinTolerance1]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/40];	# This was /20, so maybe change back?
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance1...call another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testMinTolerance1]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/80];	# This was /20, so maybe change back?
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance2...call another file for this (just to keep this file clean)
		set currentTolerance 	[expr $testMinTolerance2]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt		[expr $dtForAnalysis/10];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance3...call another file for this (just to keep this file clean)
		# Decrease dT more
		set currentTolerance 	[expr $testMinTolerance3]
		set currentNumIterations 	[expr $testHighIter]
		set currentDt		[expr $dtForAnalysis/20];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance4...call another file for this (just to keep this file clean)
		# Increase the number of iterations
		set currentTolerance 	[expr $testMinTolerance4]
		set currentNumIterations 	[expr $testHighIter]
		set currentDt		[expr $dtForAnalysis/20];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance5...call another file for this (just to keep this file clean)
		set currentTolerance 	[expr $testMinTolerance5]
		set currentNumIterations 	[expr $testHighIter]
		set currentDt		[expr $dtForAnalysis/20];
		source SolutionAlgorithmSubFile.tcl

	# Write a line to the convPlotFile to give the maximumTolerance used at this step (to plot convIndex over time)
		set convPlotFileLine "$currentTime $maxTolUsedInCurrentStep"
		puts $convPlotFilename $convPlotFileLine

	# Update the time
    		set currentTime [getTime]

	#############################################################################################
	# Now for the step check for collapse or singularity; then stop anaysis is appropriate.
		# Buffalo - You probably don't need to worry about this stuff.  It is just to try to catch a singularity error before it 
		#	make the analysis blow up!  This is to avoid numerical problems during analysis that become very nonlinear and 
		#	loose almost all stiffness.

		# Check if the model went IND (singular matrix).  If it did go IND, then call it collapsed (if useINDAsCollapse == 1), 
		#	otherwise, just let this algorithm continue and it will return an error that the displ is IND and it couldn't
		#	do the calculation below.  Note: Check for all floor nodes, as sometimes the IND or QNAN is not output to all node 
		#	recorders.  NOTICE - this may not work, in the case that you record many nodes and the floor node recorders do not get
		#	updated with QNAN of IND (I am not sure why some recorders get the singularity info, while others do not).  Note that 
		#	sometimes the recorders without the singularity information have huge displacement values in them.
			# First get some node displ responses for all floors.  
			#	In loop also check for QNAN and/or IND in the current nodal disp.  Note that for QNAN and IND, if it is
			#	-1 then it is not singular or IND.

			# ATTENTION: The items in the LISTS are added sequentially, they are not indexed by the floorNum (but in this case it works the same)

			set floorNum 1;	# initialize
			foreach nodeNumAtCurrentFloor $nodeNumsAtEachFloorLIST {
				# Find the displacement for the node at this floor and save it in the vector 
				#	(if node num is -1, we are at the ground, so set the displ equal to zero)
					if {$nodeNumAtCurrentFloor == -1} {
						# Ground floor - displ of zero.  Add this to the list
						set floorDisplVECTOR($floorNum) 0.0;
						set checkForQNANAtFlrVECTOR($floorNum) -1;
						set checkForINDAtFlrVECTOR($floorNum) -1;
					} else {
						# Not at ground floor, find displ...
						set floorDisplVECTOR($floorNum) "[nodeDisp $nodeNumAtCurrentFloor 1]"
						set checkForQNANAtFlrVECTOR($floorNum) [string first QNAN $floorDisplVECTOR($floorNum) 1];
						set checkForINDAtFlrVECTOR($floorNum) [string first IND $floorDisplVECTOR($floorNum) 1];
					}
				# Increment floor number
				set floorNum [expr $floorNum + 1];
			}

			# Also check if the node resposes are too large, b/c often sigularity is show by huge node displ's, and not just
			#	the error.  Check this at the node at each floor.
			set hugeDispTolForSing	1000000.0;
			# If any of the displ are greater than this (abs value), then call if singular.
				# Loop over all of the displacements and find the maximum displacement.  Also check the QNAN and IND at each floor and 
				#	the singularity flag if there is IND of QNAN.
				# Note that this algorithm does not use IND or QNAN as a proxy for collapse!
				set maxAbsDispl 0.0;
				set currentFloorNum 1;	# Just sent an index to loop through these arrays
				set numFloors [expr [array size floorDisplVECTOR]];
#				puts "numFloors is $numFloors"
				while {($currentFloorNum <= $numFloors)} {
					
					# Check is the displacement is greater than any previous and alter the max. if needed.
					#puts "Current floor displ: $floorDisplVECTOR($currentFloorNum)"
					if {abs($floorDisplVECTOR($currentFloorNum)) > $maxAbsDispl} {
						# Updated maximum
						set maxAbsDispl [expr abs($floorDisplVECTOR($currentFloorNum))];
					}
					
					# Check is there is IND or QNAN for this floor and if there is, set the singularity flag
					if {($checkForQNANAtFlrVECTOR($currentFloorNum) != -1) || ($checkForINDAtFlrVECTOR($currentFloorNum) != -1)} {
						# Call it singular
						set isSingularR 1;
						puts "#########################################################"
						puts "########## It's singular, so stop analysis! #############"
						puts "#########################################################"
					}

					# Updated the index nito the 
					set currentFloorNum [expr $currentFloorNum + 1];	
				}
				
				# If the maxAbsDispl is greater than the threshold, call it singular
				if {$maxAbsDispl > $hugeDispTolForSing} {
					# Call it singular
					set isSingularR 1;
					puts "#########################################################"
					puts "########## It's singular, so stop analysis! #############"
					puts "#########################################################"
				}

		# If this step is not singular, then try the normal drift check for collapse
			if {$isSingularR == 0} {
			# Determine if the structure is collapsed and if it is then stop the analysis

			# Loop and compute the drifts at each floor
			set numStories [expr $numFloors - 1];
#			puts "numStories is $numStories"
			set currentStoryNum 1;
			set floorNumBottomFloorOfStory 1;	# initialize
			while {($currentStoryNum <= $numStories)} {
				set floorNumTopFloorOfStory [expr $floorNumBottomFloorOfStory + 1];

				# Compute and save the drift for this story
#					# Checks
#					puts "floorNumBottomFloorOfStory is $floorNumBottomFloorOfStory"
#					puts "floorNumTopFloorOfStory is $floorNumTopFloorOfStory"
#					puts "Bottom floor displ: $floorDisplVECTOR($floorNumBottomFloorOfStory)"
#					puts "Top floor displ: $floorDisplVECTOR($floorNumTopFloorOfStory)"

				set storyDriftRatioVECTOR($currentStoryNum) [expr ($floorDisplVECTOR($floorNumTopFloorOfStory) - $floorDisplVECTOR($floorNumBottomFloorOfStory)) / ([lindex $floorHeightsLIST $floorNumTopFloorOfStory] - [lindex $floorHeightsLIST $floorNumBottomFloorOfStory])]

				# Save the maximum/minimum drift ratios for the full structure
				if {($storyDriftRatioVECTOR($currentStoryNum) > $maxStoryDriftRatioForFullStrR)} {
					set maxStoryDriftRatioForFullStrR [expr $storyDriftRatioVECTOR($currentStoryNum)]
				}
				if {($storyDriftRatioVECTOR($currentStoryNum) < $minStoryDriftRatioForFullStrR)} {
					set minStoryDriftRatioForFullStrR [expr $storyDriftRatioVECTOR($currentStoryNum)]
				}

				# If the story drift is larger than the tolerance, set the flag and call it collapsed!
				if {($maxStoryDriftRatioForFullStrR > $minStoryDriftRatioForCollapse ) || ($minStoryDriftRatioForFullStrR < [expr -($minStoryDriftRatioForCollapse)])} {
					# Changed the flag to make the EQ analysis stop
					set isCollapsedR 1
					puts "########################################################"
					puts "########## It collapsed, so stop analysis! #############"
					puts "########################################################"
				}				

				# Increment floor and story number
				set floorNumBottomFloorOfStory [expr $floorNumBottomFloorOfStory + 1];
				set currentStoryNum [expr $currentStoryNum + 1];
			}

			#puts "Current maxStoryDriftRatioForFullStrR is $maxStoryDriftRatioForFullStrR"
			#puts "Current minStoryDriftRatioForFullStrR is $minStoryDriftRatioForFullStrR"

		# End of checking for collapse and singularity for this step
		#############################################################################################

	}; # I am not sure why we have two brackets here, but it makes it work!
	}; # I am not sure why we have two brackets here, but it makes it work!

	remove loadPattern $tag
	loadConst -time 0.0

#############################################################################################
# Compute if the EQ is fully converged and output this to the file (output is done below)
	set timeTolerance 2.0;	# We will call it converged as long as it got to within 2.0 seconds of the end of the EQ (in case there is some numerical rounding).
	if {$currentTime > ($maxTime - $timeTolerance)} {
		puts "EQ fully converged"
		set isNonConvR 0
	} else {
		puts "EQ NOT FULLY converged"
		set isNonConvR 1
	}	

# Output the convergence information to the convergence file
	
		set startDir [pwd]
		cd ..
		cd ..
			
		set baseDir [pwd]
		cd $baseDir/Output/$analysisType/EQ_$eqNumber/Sa_$saTOneForRun
		set runDir [pwd]

		cd $runDir/RunInformation/

		set convFilename [open maxTolUsedOUT.out w]
		puts $convFilename $maxTolUsed
		close $convFilename 

		set tempFile [open isNonConvOUT.out w]
		puts $tempFile $isNonConvR 
		close $tempFile 

		cd $startDir 

# Output a final statement and close the file stream for the convergence log
		# Write a line in the convergence log telling that we are finished and fully converged
		set convFileLine "End of the EQ!"
		puts $convLogFilename $convFileLine 

		close $convLogFilename 


##########################################################################
##########################################################################
############### Older Algorithms #########################################
##########################################################################
##########################################################################

} elseif {$EQsolutionAlgorithm == "CordovaMod"} {
	puts "EQ Solution Algorithm: CordovaMod"
	Don't use this - this has not been updated with the new collapse algo to test for non-conv (see notes on 1-28-05)

	# I changed this to Line Search (it was just Newton) to see if it will make it work (on 6-30-04)
	#     Using the Newton for the first step made the model run for Paul, but it
	#	made my model (when I added the gravity frames) not work at all).  Now 
	#	that I am using NewtonLineSearch from the start, then it's working!!!
	#algorithm Newton
	#algorithm NewtonLineSearch 0.6
	#algorithm NewtonLineSearch 0.8
	#algorithm Newton -initial
	#algorithm KrylovNewton 
	#algorithm KrylovNewton -initial

	system UmfPack

	# Define integrator
	integrator Newmark 0.5  0.25 

	# Define damping
#	rayleigh  $alpha1 $alpha2 0 0; 	# Current stiffness
	rayleigh   $alpha1 0 $alpha2 0;	# Intial stiffness

	# DOF numberer
	numberer RCM

	# Constraint handler
#	constraints Transformation
	constraints $constraintForEQ $constraintArg1EQ $constraintArg2EQ
#	constraints Penalty 1.0e15 1.0e15

	# create the analysis object
	analysis Transient

	# I took this out, so all of the steps are in the analysis loop
	#analyze 1 $dtForAnalysis;	# Paul took one step with Newton when he did his model.

	# switch to NewtonLineSearch - [[ Paul's old comment - "for some reason this does not work 
	# on the very first step, but fine for all succeeding steps"]].

	# Changed here with Frank on 7-6-04 (he reocmended using Newton instead of LineSearch)
	algorithm $iterAlgo

	#Analysis loop - this wil complete when the EQ is over, the structure collapses, or the system goes singular
	while {$ok == 0 && $currentTime < $maxTime && $isCollapsedR == 0 && $isSingularR == 0} {

      	puts "tFinal is $maxTime; and tCurrent is $currentTime"

		# Do step with initial tolerance and input algorithm
		test $testType $testTolerance $testIterations 0
    		set ok [analyze 1 $dtForAnalysis]

		# Keep track of the maximum tolarance used in this step  - for the convPlotFile.  This is later increased it necessary.
		set maxTolUsedInCurrentStep 	[expr $testTolerance]

		#### Change things for convergence ###############
		# If it's not ok, try to decrease dT, but keep the toerance the samecall another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testTolerance]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/10];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, try to decrease dT a bit more, but keep the toerance the samecall another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testTolerance]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/20];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, try to decrease dT a bit more, but keep the toerance the samecall another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testTolerance]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/40];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, try to decrease dT a bit more, but keep the toerance the samecall another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testTolerance]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/80];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance1...call another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testMinTolerance1]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/10];	# This was /20, so maybe change back?
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance1...call another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testMinTolerance1]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/20];	# This was /20, so maybe change back?
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance1...call another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testMinTolerance1]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/40];	# This was /20, so maybe change back?
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance1...call another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testMinTolerance1]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/80];	# This was /20, so maybe change back?
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance2...call another file for this (just to keep this file clean)
		set currentTolerance 	[expr $testMinTolerance2]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt		[expr $dtForAnalysis/10];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance3...call another file for this (just to keep this file clean)
		# Decrease dT more
		set currentTolerance 	[expr $testMinTolerance3]
		set currentNumIterations 	[expr $testHighIter]
		set currentDt		[expr $dtForAnalysis/20];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance4...call another file for this (just to keep this file clean)
		# Increase the number of iterations
		set currentTolerance 	[expr $testMinTolerance4]
		set currentNumIterations 	[expr $testHighIter]
		set currentDt		[expr $dtForAnalysis/20];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance5...call another file for this (just to keep this file clean)
		set currentTolerance 	[expr $testMinTolerance5]
		set currentNumIterations 	[expr $testHighIter]
		set currentDt		[expr $dtForAnalysis/20];
		source SolutionAlgorithmSubFile.tcl

	# Write a line to the convPlotFile to give the maximumTolerance used at this step (to plot convIndex over time)
		set convPlotFileLine "$currentTime $maxTolUsedInCurrentStep"
		puts $convPlotFilename $convPlotFileLine

	# Update the time
    		set currentTime [getTime]

	# Check if the model went IND (singular matrix).  If it did go IND, then call it collapsed (if useINDAsCollapse == 1), 
	#	otherwise, just let this algorithm continue and it will return an error that the displ is IND and it couldn't
	#	do the calculation below.  Note: Check for all floor nodes, as sometimes the IND or QNAN is not output to all node 
	#	recorders.  NOTICE - this may not work, in the case that you record many nodes and the floor node recorders do not get
	#	updated with QNAN of IND (I am not sure why some recorders get the singularity info, while others do not).  Note that 
	#	sometimes the recorders without the singularity information have huge displacement values in them.
		# First get some node displ responses
			set floor2Displ "[nodeDisp $nodeNumAtFloor2 1]"
			set floor3Displ "[nodeDisp $nodeNumAtFloor3 1]"
			set floor4Displ "[nodeDisp $nodeNumAtFloor4 1]"
			set floor5Displ "[nodeDisp $nodeNumAtFloor5 1]"

			set checkForQNANFlr2 	[string first QNAN $floor2Displ 1]
			set checkForINDFlr2 	[string first IND $floor2Displ 1]
			set checkForQNANFlr3 	[string first QNAN $floor3Displ 1]
			set checkForINDFlr3 	[string first IND $floor3Displ 1]
			set checkForQNANFlr4 	[string first QNAN $floor4Displ 1]
			set checkForINDFlr4 	[string first IND $floor4Displ 1]
			set checkForQNANFlr5 	[string first QNAN $floor5Displ 1]
			set checkForINDFlr5 	[string first IND $floor5Displ 1]

			# Also check if the node resposes are too large, b/c often sigularity is show by huge node displ's, and not just
			#	the error.
			set hugeDispTolForSing	1000000.0;
			# If any of the displ are greater than this (abs value), then call if singular.
			if {($floor2Displ > $hugeDispTolForSing) || ($floor2Displ < -$hugeDispTolForSing) || ($floor3Displ > $hugeDispTolForSing) || ($floor3Displ < -$hugeDispTolForSing) || ($floor4Displ > $hugeDispTolForSing) || ($floor4Displ < -$hugeDispTolForSing) || ($floor5Displ > $hugeDispTolForSing) || ($floor5Displ < -$hugeDispTolForSing)} {
				# Call it singular
				set isSingularR 1;
			}
						
		# Now, if we are using QNAN for collapse, check for "collapse"
		# Check to see if the displacement became QNAN.  If it did, then call it collapsed and stop analysis.
		if {($checkForQNANFlr2 != -1) || ($checkForINDFlr2 != -1) || ($checkForQNANFlr3 != -1) || ($checkForINDFlr3 != -1) || ($checkForQNANFlr4 != -1) || ($checkForINDFlr4 != -1) || ($checkForQNANFlr5 != -1) || ($checkForINDFlr5 != -1)} {
			# Set the flag to show that it became singular
			set isSingularR 1;
			# Now, if we are using QNAN for collapse, then call it collapsed
			if {$useINDAsCollapse == 1} {
			# Check to see if the displacement became QNAN.  If it did, then call it collapsed and stop analysis.
				puts "COLLASPED - It worked - due to QNAN or IND we called it COLLAPSED!"
				# Indeterminate, call it collapsed!
				set isCollapsedR 1
			}
		}

	# If this step is not singular, then try the normal drift check for collapse
		if {$isSingularR == 0} {
			# Determine if the structure is collapsed and if it is then stop the analysis
				set floor2Displ [nodeDisp $nodeNumAtFloor2 1]
				set floor3Displ [nodeDisp $nodeNumAtFloor3 1]
				set floor4Displ [nodeDisp $nodeNumAtFloor4 1]
				set floor5Displ [nodeDisp $nodeNumAtFloor5 1]

				# Compute drift ratios with drift divided by story heights (from the list)
				set story1DriftRatio [expr ($floor2Displ) / [lindex $floorHeightsLIST 1]]
				set story2DriftRatio [expr ($floor3Displ - $floor2Displ) / [lindex $floorHeightsLIST 2]]
				set story3DriftRatio [expr ($floor4Displ - $floor3Displ) / [lindex $floorHeightsLIST 3]]
				set story4DriftRatio [expr ($floor5Displ - $floor4Displ) / [lindex $floorHeightsLIST 4]]

				# Update the max/min drifts for full structure
					if {($story1DriftRatio > $maxStoryDriftRatioForFullStrR)} {
						set maxStoryDriftRatioForFullStr [expr $story1DriftRatio]
					}
					if {($story2DriftRatio > $maxStoryDriftRatioForFullStrR)} {
						set maxStoryDriftRatioForFullStrR [expr $story2DriftRatio]
					}
					if {($story3DriftRatio > $maxStoryDriftRatioForFullStrR)} {
						set maxStoryDriftRatioForFullStrR [expr $story3DriftRatio]
					}
					if {($story4DriftRatio > $maxStoryDriftRatioForFullStrR)} {
						set maxStoryDriftRatioForFullStrR [expr $story4DriftRatio]
					}
					if {($story1DriftRatio < $minStoryDriftRatioForFullStrR)} {
						set minStoryDriftRatioForFullStrR [expr $story1DriftRatio]
					}
					if {($story2DriftRatio < $minStoryDriftRatioForFullStrR)} {
						set minStoryDriftRatioForFullStrR [expr $story2DriftRatio]
					}
					if {($story3DriftRatio < $minStoryDriftRatioForFullStrR)} {
						set minStoryDriftRatioForFullStrR [expr $story3DriftRatio]
					}
					if {($story4DriftRatio < $minStoryDriftRatioForFullStrR)} {
						set minStoryDriftRatioForFullStrR [expr $story4DriftRatio]
					}


		# Checks
#		puts "maxStoryDriftRatioForFullStrR is: $maxStoryDriftRatioForFullStrR"
#		puts "minStoryDriftRatioForFullStrR is: $minStoryDriftRatioForFullStrR"
#
				# Find if it's collapsed - if any floor has too high of a story drift
				if {($maxStoryDriftRatioForFullStrR > $minStoryDriftRatioForCollapse ) || ($minStoryDriftRatioForFullStrR < [expr -($minStoryDriftRatioForCollapse)])} {
					# Changed the flag to make the EQ analysis stop
					set isCollapsedR 1

					# NOT USED...
#					# Now define the variable in the scope of the file that called this function, effectively returning the isCollapsed variable
#					#	to the calling function (Tcl text page 138)
#					uplevel "set isCollapsedForCurrentRun [expr $isCollapsed]"
				}
		}
	}
	remove loadPattern $tag
	loadConst -time 0.0


# Output the convergence information to the convergence file
	
		set startDir [pwd]
		cd ..
		cd ..
			
		set baseDir [pwd]
		cd $baseDir/Output/$analysisType/EQ_$eqNumber/Sa_$saTOneForRun
		set runDir [pwd]

		cd $runDir/RunInformation/

		set convFilename [open maxTolUsedOUT.out w]
		puts $convFilename $maxTolUsed
		close $convFilename 

		cd $startDir 

# Output a final statement and close the file stream for the convergence log
		# Write a line in the convergence log telling that we are finished and fully converged
		set convFileLine "End of the EQ!"
		puts $convLogFilename $convFileLine 

		close $convLogFilename 

##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################

} else {
	puts "ERROR: No EQ solution algoritm used - check analysis options" 
	ERROR
}

# Output file saying whether or not it used NormDisplIncr
			set startDir [pwd]
			cd ..
			cd ..
			
			set baseDir [pwd]
			cd $baseDir/Output/$analysisType/EQ_$eqNumber/Sa_$saTOneForRun
			set runDir [pwd]

			file mkdir RunInformation; 
			cd $runDir/RunInformation/

			set filenameEQ [open usedNormDisplIncrOUT.out w]
			puts $filenameEQ $usedNormDisplIncr
			close $filenameEQ 

			# Output the scaled Sa,comp.  This is from Matlab now (6-29-06).
			set filename [open saCompScaledOUT.out w]
			puts $filename $saCompScaled
			close $filename 

			# Output the scaled Sa,geoMean.  This comes from Matlab now (6-29-06) and not that it is =-1 if we sacled the 
			#	GMs by Sa,comp (b/c when scaling by Sa,comp we likely did not define both GM components) (6-29-06)
			set filename [open saGeoMeanScaledOUT.out w]
			puts $filename $saGeoMeanScaled
			close $filename 

			# Output the scale factor - This was sent from Matlab and is the factor applied to the component (6-29-06)
			set filename [open scaleFactorAppliedToCompOUT.out w]
			puts $filename $scaleFactorForRunFromMatlab
			close $filename 

			# Output the file format and scaling mthod used
			set filename [open eqFormatForCollapseListOUT.out w]
			puts $filename $eqFormatForCollapseList
			close $filename 

			cd $startDir 

# Return if it collapsed or not - not anymore, as I am using "upvar"
#return $isCollapsed 

}

#####################################################################################################
#####################################################################################################
####################################################################################################
####################################################################################################

#####################################################################################################
#####################################################################################################
####################################################################################################
####################################################################################################

proc RunEQLoadingForCollapse_withScaleFactor {eqNumber eqFileName scaleFactorForRunFromMatlab SF_DirectoryName dtForAnalysis alpha1 alpha2 g testType allowNormDisplIncrForConv analysisType EQsolutionAlgorithm iterAlgo iterAlgoArg nodeNumsAtEachFloorLIST minStoryDriftRatioForCollapse useINDAsCollapse eqFormatForCollapseList floorHeightsLIST isCollapsed isSingular isNonConv maxStoryDriftRatioForFullStr minStoryDriftRatioForFullStr constraintForEQ constraintArg1EQ constraintArg2EQ listOfElementsForRayleighDamping } {


# ** This procedure was modified from "RunEQLoadingForCollapse" to run for a set scale factor instead of a set Sa value (CBH 7-24-07).
# This does not use the DefineRecordInformation.tcl file (does not need Sa values and does.


# This is just like RunEQLoading, except the algorithm stops if collapse is detected (based on input into SetAnalysisOptions).
#	This algorithm is currently specific to ONLY A FOUR STORY STRUCTURE (due to the checking for collapse and singularity
#	at each step of the analysis)!

# eqFileName - specifies name of the record (file name) to be run - this file must be in the EQs folder
# alpha1, alpha2 - the damping coefficients
# scaleFactor - the scaling factor on the record (ex. 1, 1.5, 2)
# dT - the time step used for integration
# numPointsForEQ - the number of points for the eq record
# g - gravitational constant
# testType - convergence test to be used - e.g. normDisplIncr
# allowNormDisplIncrForConv - tells whther or not NormDisplIncr is allowed if convergence is bad - used below in solution algorithm
# analysisType - this is just used for a folder name for outputing whether NormDisplIncr was used or not.
# EQsolutionAlgorithm - the solution algorithm to use
# displayMode - if the display is on or off
###################################################################################################

# First define the variables that will be passed back to the calling function using upvar...
	# Buffalo - you don't need to knoww hat this is doing.
	# These variables are used to try to find in the building is collapse and/or singular at each step of the analysis.
	# The "R" simply means that it is a returned value
	upvar isCollapsed isCollapsedR;
	upvar isSingular isSingularR;
	upvar isNonConv isNonConvR;
	upvar maxStoryDriftRatioForFullStr maxStoryDriftRatioForFullStrR;
	upvar minStoryDriftRatioForFullStr minStoryDriftRatioForFullStrR;

#################################
# Make the directory for RunInformation, to output convergence information into.  Also open a file stream for the convergence log
		set startDir [pwd]
		cd ..
		cd ..
			
		set baseDir [pwd]
		puts "Check: when trying to open the SF directory, SF_DirectoryName is $SF_DirectoryName (in proc RunEQLoadingForCollapse_withScaleFactor)"
		cd $baseDir/Output/$analysisType/EQ_$eqNumber/$SF_DirectoryName
		set runDir [pwd]

		file mkdir RunInformation; 
		cd $runDir/RunInformation/

		set convLogFilename 	[open convLogFileOUT.out w]
		set convPlotFilename	[open convPlotFileOUT.out w]

		cd $startDir 
##################################

#puts "EQ number is: $eqNumber"
#puts "Current directory is: [pwd]"

# Source in the information for the EQ's
# NOT USED NOW - source DefineEarthquakeRecordInformation.tcl

## The scale factor is now computed in Matlab and sent to Opensees and this function; therefore, we do nto need to compute it anymore!! (6-29-06)
#if {$eqFormatForCollapseList == "Formatted_singleComp" || $eqFormatForCollapseList == "PEER_singleComp" || $eqFormatForCollapseList == "PEER-NGA_singleComp" || $eqFormatForCollapseList == "PEER-NGA_Rotated_singleComp"} { 
#	set scaleFactor [expr $saTOneForRun / $saTOneForEQ($eqNumber)]
#} elseif {$eqFormatForCollapseList == "Formatted_geoMean" || $eqFormatForCollapseList == "PEER_geoMean" || $eqFormatForCollapseList == "PEER-NGA_geoMean" || $eqFormatForCollapseList == "PEER-NGA_Rotated_geoMean"} {
#	# The Eq number that we are running is for a single component, so we need to 
#	# 	compute the number that I used that is generic to both component (the number that is about 10x less)
#	# 	and get the GeoMean with that number.
#	set eqNumberForGeoMean [expr $eqNumber / 10]
#	set scaleFactor [expr $saTOneForRun / $saTOneForEQGeoMean($eqNumberForGeoMean)]
#	puts "eqNumberForGeoMean is $eqNumberForGeoMean"
#	puts "scaleFactor is $scaleFactor"
#
#} elseif {$eqFormatForCollapseList == "Formatted_codeScaling" || $eqFormatForCollapseList == "PEER_codeScaling" || $eqFormatForCollapseList == "PEER-NGA_codeScaling" || $eqFormatForCollapseList == "PEER-NGA_Rotated_codeScaling"} {
#	# The Eq number that we are running is for a single component, so we need to 
#	# 	compute the number that I used that is generic to both component (the number that is about 10x less)
#	# 	and get the GeoMean with that number.
#	set eqNumberForCodeScaling [expr $eqNumber / 10]
#	set scaleFactor [expr ($saTOneForRun * $scaleFactorForMeanOfCodeGMSetToBeOne($eqNumberForCodeScaling))]
#	puts "eqNumberForCodeScaling is $eqNumberForCodeScaling";
#	puts "scaleFactor is $scaleFactor"
#
#} else {
#	puts "ERROR - eqRecordListToUse not found!"
#	ERROR
#}
#

# Output some stuff
	puts "Running EQ for: EQ $eqNumber at scale factor of $scaleFactorForRunFromMatlab"
	#puts "saTOneForRun is $saTOneForRun"
	#puts "scaleFactorForRunFromMatlab is $scaleFactorForRunFromMatlab"
	#puts "saCompScaled is $saCompScaled"
	#puts "saGeoMeanScaled is $saGeoMeanScaled"

# Initialize variables used for the analysis
	set usedNormDisplIncr 			0;			# Did not yet need to use NormDisplIncr to converge
	set isCollapsedR 				0;			# Not yet collapsed
	set isSingularR 				0;			# Not yet singular
	set isNonConvR 				0;			# Not yet non-converged
	set maxStoryDriftRatioForFullStrR 	0;			# No drift yet
	set minStoryDriftRatioForFullStrR 	0;			# No drift yet
	set isCollapsedForCurrentRun 0;				# Initialize - I am not sure if I need this or not

# Depending on what type of record format is being used, then do different things to set up the record to run...
	if {$eqFormatForCollapseList == "Formatted_singleComp" } {
		# Option added by CBH on 9-25-07
		# We need to read the dt and numPoints files that have been pre-saved in the EQ folder.

			# Load dt from the file
			set fileID [open C:/OpenSeesProcessingFiles/EQs/DtFile_($eqNumber).txt r]
			gets $fileID dt
			puts "dt is $dt"
			close $fileID

			# Load numPoints from the file (note: I do not think this is actually used in the analysis procedure, but I am not positive)
			set fileID [open C:/OpenSeesProcessingFiles/EQs/NumPointsFile_($eqNumber).txt r]
			gets $fileID numPoints 
			puts "numPoints is $numPoints"
			close $fileID

		# Define the serie using the pre-saved SortedEQFile
		set Series "Path -filePath C:/OpenSeesProcessingFiles/EQs/SortedEQFile_($eqNumber).txt -dt $dt -factor [expr $scaleFactorForRunFromMatlab * $g]"

	} elseif {$eqFormatForCollapseList == "PEER-NGA_singleComp" || $eqFormatForCollapseList == "PEER-NGA_Rotated_singleComp" } {
		# Run PEER formatted record
		###########
		# Use the procedure to get dt and numPoints for the EQ and make a file for the EQ that has a single column of accel TH in it.
			# Initialize the dt and numPoints for the current EQ.  These are passed-by-reference to the procedure, so the proc
			#	changes thier values.
			set dt 		0.0;	# This will be changed by ReadSMDFile2 (pass-by-reference)
			set numPoints	0;	# This will be changed by ReadSMDFile2 (pass-by-reference)
			
			# Call the procedure to do the "pre-processing"; depends on file format
			if { $eqFormatForCollapseList == "PEER-NGA_singleComp" } {
				# The EQ file name is now passed into this procedure (the name comes from Matlab, is written
				#	to a TCL script, read, then is passed this procedure .  The EQ Information file is 
				#	no longer used.
				set eqFileNameForProc "C:/OpenSeesProcessingFiles/EQs/A_PEERNGADatabase/$eqFileName"
				set outputFileNameForProc "C:/OpenSeesProcessingFiles/EQs/SortedEQFile_($eqNumber).txt"

				# Call the file that processes the records in the new PEER-NGA format
				ReadSMDFile_PEER-NGA-Format $eqFileNameForProc $outputFileNameForProc $dt $numPoints
			} elseif { $eqFormatForCollapseList == "PEER-NGA_Rotated_singleComp" } {
				# The EQ file name is now passed into this procedure (the name comes from Matlab, is written
				#	to a TCL script, read, then is passed this procedure .  The EQ Information file is 
				#	no longer used.
				set eqFileNameForProc "C:/OpenSeesProcessingFiles/EQs/A_PEERNGADatabase_Rotated/$eqFileName"
				set outputFileNameForProc "C:/OpenSeesProcessingFiles/EQs/SortedEQFile_($eqNumber).txt"

				# Call the file that processes the records in the new PEER-NGA format
				ReadSMDFile_PEER-NGA-Rotated-Format $eqFileNameForProc $outputFileNameForProc $dt $numPoints
			} else {
				puts "ERROR - file format input not valid!"
				ERROR - stop analysis
			}

#			puts "dt is $dt"
#			puts "numPoints is $numPoints"

		###########

		set Series "Path -filePath C:/OpenSeesProcessingFiles/EQs/SortedEQFile_($eqNumber).txt -dt $dt -factor [expr $scaleFactorForRunFromMatlab * $g]"
	} else {
		puts "eqFormatForCollapseList is $eqFormatForCollapseList"
		puts "ERROR: The EQ record format is not defined correctly!"
		ERROR - stop analysis
	}


# Set the tab number for the EQ pattern
	set tag 101

# Define the loading patern for the EQ
	#                           tag  dir  accel    series
	pattern UniformExcitation  $tag   1  -accel   $Series

# Initial variables before starting dynamic analysis
	set ok 0
	set currentTime [getTime]
	set maxTime [expr [getTime]+[expr ($numPoints)*$dt]];  
	set numSteps [expr 1 * [expr ($numPoints)*$dt] / [expr ($dtForAnalysis)]]

# Define the ranges of tolerances to try for convergence.  Note that the maximum tolerance 
#	used in the analsyis will be output in the output files.
	set testTolerance 1.0e-6
	set testMinTolerance1 1.0e-5;	
	set testMinTolerance2 1.0e-4;	
	set testMinTolerance3 1.0e-3;	
	set testMinTolerance4 1.0e-2;	
	set testMinTolerance5 1.0e-1;	
		
# Define the iteration information used for difference situations and tests
	set testIterations 100;			# Changed on 1-7-05 for Corotational transformation
	set testInitialIterations 1000
	set testLowIter 10;	# Used to try each test in the loop
	set ratioForInitialAlgo	200;	# This is the ratio of testIterations that is allowed for -initial test 
	set testHighIter 1000;	# Used to try to make it converge at the very end
	
# Define the initial test, this will be changed later during convergance work.
	test $testType $testTolerance $testIterations 0

# Set initialize the maximum tolerance used - for output to know about what tolerance was used to obtain convergence.
	set maxTolUsed $testTolerance;


##########################################################################
# Decide which solution algoritm to use - based on the analysis options

if {$EQsolutionAlgorithm == "CordovaFrank"} {
	puts "EQ Solution Algorithm: CordovaFrank"

	# Use this alogorithm!  This is the best one!

	# This is the same as CordovaMod, except that I use Franks adaptative proceedure when calling the "analyze" command.

	# I changed this to Line Search (it was just Newton) to see if it will make it work (on 6-30-04)
	#     Using the Newton for the first step made the model run for Paul, but it
	#	made my model (when I added the gravity frames) not work at all).  Now 
	#	that I am using NewtonLineSearch from the start, then it's working!!!
	#algorithm Newton
	#algorithm NewtonLineSearch 0.6
	#algorithm NewtonLineSearch 0.8
	#algorithm Newton -initial
	#algorithm KrylovNewton 
	#algorithm KrylovNewton -initial


	##########
	# Set options to use Franks adaptative analysis
		#analyze $numIncr <$dt> <$dtMin $dtMax $Jd>
	set numIncrForFranksAnalysis	1;	# This is the number of increments that are used before I enter my convergence loop 
							# BE SURE TO DO 1 so it will pop out and I can check the displ. for possible 
							# 	collapse and for singularity AT EVERY STEP (changed on 1-21-05).
	set dtFrank				[expr $dtForAnalysis];
	set dtMinFrank			[expr $dtForAnalysis / 100.0];
	set dtMaxFrank			[expr $dtForAnalysis];
	##########

	system UmfPack

	# Define integrator
	integrator Newmark 0.5  0.25 

	# Define damping - this is done with VB now
	#source DefineDampingObjects.tcl

	# DOF numberer
	numberer RCM

	# Constraint handler
#	constraints Transformation
	constraints $constraintForEQ $constraintArg1EQ $constraintArg2EQ
#	constraints Penalty 1.0e15 1.0e15

	# Create the analysis object
	analysis Transient

	# I took this out, so all of the steps are in the analysis loop
	#analyze 1 $dtForAnalysis;	# Paul took one step with Newton when he did his model.

	# switch to NewtonLineSearch - [[ Paul's old comment - "for some reason this does not work 
	# on the very first step, but fine for all succeeding steps"]].

	# Changed here with Frank on 7-6-04 (he reocmended using Newton instead of LineSearch)
	algorithm $iterAlgo

	#Analysis loop - this wil complete when the EQ is over, the structure collapses, or the system goes singular
	while {$ok == 0 && $currentTime < $maxTime && $isCollapsedR == 0 && $isSingularR == 0} {

      	puts "tFinal is $maxTime; and tCurrent is $currentTime"

		# Do step with initial tolerance and input algorithm
		test $testType $testTolerance $testIterations 0


		# Altered here for Cordova Frank
#    		set ok [analyze 1 $dtForAnalysis]
		set ok [analyze $numIncrForFranksAnalysis $dtFrank $dtMinFrank $dtMaxFrank]


		# Keep track of the maximum tolarance used in this step  - for the convPlotFile.  This is later increased it necessary.
		set maxTolUsedInCurrentStep 	[expr $testTolerance]

		#### Change things for convergence ###############
		# If it's not ok, try to decrease dT, but keep the toerance the samecall another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testTolerance]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/10];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, try to decrease dT a bit more, but keep the toerance the samecall another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testTolerance]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/20];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, try to decrease dT a bit more, but keep the toerance the samecall another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testTolerance]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/40];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, try to decrease dT a bit more, but keep the toerance the samecall another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testTolerance]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/80];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance1...call another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testMinTolerance1]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/10];	# This was /20, so maybe change back?
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance1...call another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testMinTolerance1]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/20];	# This was /20, so maybe change back?
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance1...call another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testMinTolerance1]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/40];	# This was /20, so maybe change back?
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance1...call another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testMinTolerance1]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/80];	# This was /20, so maybe change back?
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance2...call another file for this (just to keep this file clean)
		set currentTolerance 	[expr $testMinTolerance2]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt		[expr $dtForAnalysis/10];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance3...call another file for this (just to keep this file clean)
		# Decrease dT more
		set currentTolerance 	[expr $testMinTolerance3]
		set currentNumIterations 	[expr $testHighIter]
		set currentDt		[expr $dtForAnalysis/20];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance4...call another file for this (just to keep this file clean)
		# Increase the number of iterations
		set currentTolerance 	[expr $testMinTolerance4]
		set currentNumIterations 	[expr $testHighIter]
		set currentDt		[expr $dtForAnalysis/20];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance5...call another file for this (just to keep this file clean)
		set currentTolerance 	[expr $testMinTolerance5]
		set currentNumIterations 	[expr $testHighIter]
		set currentDt		[expr $dtForAnalysis/20];
		source SolutionAlgorithmSubFile.tcl

	# Write a line to the convPlotFile to give the maximumTolerance used at this step (to plot convIndex over time)
		set convPlotFileLine "$currentTime $maxTolUsedInCurrentStep"
		puts $convPlotFilename $convPlotFileLine

	# Update the time
    		set currentTime [getTime]

	#############################################################################################
	# Now for the step check for collapse or singularity; then stop anaysis is appropriate.
		# Buffalo - You probably don't need to worry about this stuff.  It is just to try to catch a singularity error before it 
		#	make the analysis blow up!  This is to avoid numerical problems during analysis that become very nonlinear and 
		#	loose almost all stiffness.

		# Check if the model went IND (singular matrix).  If it did go IND, then call it collapsed (if useINDAsCollapse == 1), 
		#	otherwise, just let this algorithm continue and it will return an error that the displ is IND and it couldn't
		#	do the calculation below.  Note: Check for all floor nodes, as sometimes the IND or QNAN is not output to all node 
		#	recorders.  NOTICE - this may not work, in the case that you record many nodes and the floor node recorders do not get
		#	updated with QNAN of IND (I am not sure why some recorders get the singularity info, while others do not).  Note that 
		#	sometimes the recorders without the singularity information have huge displacement values in them.
			# First get some node displ responses for all floors.  
			#	In loop also check for QNAN and/or IND in the current nodal disp.  Note that for QNAN and IND, if it is
			#	-1 then it is not singular or IND.

			# ATTENTION: The items in the LISTS are added sequentially, they are not indexed by the floorNum (but in this case it works the same)

			set floorNum 1;	# initialize
			foreach nodeNumAtCurrentFloor $nodeNumsAtEachFloorLIST {
				# Find the displacement for the node at this floor and save it in the vector 
				#	(if node num is -1, we are at the ground, so set the displ equal to zero)
					if {$nodeNumAtCurrentFloor == -1} {
						# Ground floor - displ of zero.  Add this to the list
						set floorDisplVECTOR($floorNum) 0.0;
						set checkForQNANAtFlrVECTOR($floorNum) -1;
						set checkForINDAtFlrVECTOR($floorNum) -1;
					} else {
						# Not at ground floor, find displ...
						set floorDisplVECTOR($floorNum) "[nodeDisp $nodeNumAtCurrentFloor 1]"
						set checkForQNANAtFlrVECTOR($floorNum) [string first QNAN $floorDisplVECTOR($floorNum) 1];
						set checkForINDAtFlrVECTOR($floorNum) [string first IND $floorDisplVECTOR($floorNum) 1];
					}
				# Increment floor number
				set floorNum [expr $floorNum + 1];
			}

			# Also check if the node resposes are too large, b/c often sigularity is show by huge node displ's, and not just
			#	the error.  Check this at the node at each floor.
			set hugeDispTolForSing	1000000.0;
			# If any of the displ are greater than this (abs value), then call if singular.
				# Loop over all of the displacements and find the maximum displacement.  Also check the QNAN and IND at each floor and 
				#	the singularity flag if there is IND of QNAN.
				# Note that this algorithm does not use IND or QNAN as a proxy for collapse!
				set maxAbsDispl 0.0;
				set currentFloorNum 1;	# Just sent an index to loop through these arrays
				set numFloors [expr [array size floorDisplVECTOR]];
#				puts "numFloors is $numFloors"
				while {($currentFloorNum <= $numFloors)} {
					
					# Check is the displacement is greater than any previous and alter the max. if needed.
					#puts "Current floor displ: $floorDisplVECTOR($currentFloorNum)"
					if {abs($floorDisplVECTOR($currentFloorNum)) > $maxAbsDispl} {
						# Updated maximum
						set maxAbsDispl [expr abs($floorDisplVECTOR($currentFloorNum))];
					}
					
					# Check is there is IND or QNAN for this floor and if there is, set the singularity flag
					if {($checkForQNANAtFlrVECTOR($currentFloorNum) != -1) || ($checkForINDAtFlrVECTOR($currentFloorNum) != -1)} {
						# Call it singular
						set isSingularR 1;
						puts "#########################################################"
						puts "########## It's singular, so stop analysis! #############"
						puts "#########################################################"
					}

					# Updated the index nito the 
					set currentFloorNum [expr $currentFloorNum + 1];	
				}
				
				# If the maxAbsDispl is greater than the threshold, call it singular
				if {$maxAbsDispl > $hugeDispTolForSing} {
					# Call it singular
					set isSingularR 1;
					puts "#########################################################"
					puts "########## It's singular, so stop analysis! #############"
					puts "#########################################################"
				}

		# If this step is not singular, then try the normal drift check for collapse
			if {$isSingularR == 0} {
			# Determine if the structure is collapsed and if it is then stop the analysis

			# Loop and compute the drifts at each floor
			set numStories [expr $numFloors - 1];
#			puts "numStories is $numStories"
			set currentStoryNum 1;
			set floorNumBottomFloorOfStory 1;	# initialize
			while {($currentStoryNum <= $numStories)} {
				set floorNumTopFloorOfStory [expr $floorNumBottomFloorOfStory + 1];

				# Compute and save the drift for this story
#					# Checks
#					puts "floorNumBottomFloorOfStory is $floorNumBottomFloorOfStory"
#					puts "floorNumTopFloorOfStory is $floorNumTopFloorOfStory"
#					puts "Bottom floor displ: $floorDisplVECTOR($floorNumBottomFloorOfStory)"
#					puts "Top floor displ: $floorDisplVECTOR($floorNumTopFloorOfStory)"

				set storyDriftRatioVECTOR($currentStoryNum) [expr ($floorDisplVECTOR($floorNumTopFloorOfStory) - $floorDisplVECTOR($floorNumBottomFloorOfStory)) / ([lindex $floorHeightsLIST $floorNumTopFloorOfStory] - [lindex $floorHeightsLIST $floorNumBottomFloorOfStory])]

				# Save the maximum/minimum drift ratios for the full structure
				if {($storyDriftRatioVECTOR($currentStoryNum) > $maxStoryDriftRatioForFullStrR)} {
					set maxStoryDriftRatioForFullStrR [expr $storyDriftRatioVECTOR($currentStoryNum)]
				}
				if {($storyDriftRatioVECTOR($currentStoryNum) < $minStoryDriftRatioForFullStrR)} {
					set minStoryDriftRatioForFullStrR [expr $storyDriftRatioVECTOR($currentStoryNum)]
				}

				# If the story drift is larger than the tolerance, set the flag and call it collapsed!
				if {($maxStoryDriftRatioForFullStrR > $minStoryDriftRatioForCollapse ) || ($minStoryDriftRatioForFullStrR < [expr -($minStoryDriftRatioForCollapse)])} {
					# Changed the flag to make the EQ analysis stop
					set isCollapsedR 1
					puts "########################################################"
					puts "########## It collapsed, so stop analysis! #############"
					puts "########################################################"
				}				

				# Increment floor and story number
				set floorNumBottomFloorOfStory [expr $floorNumBottomFloorOfStory + 1];
				set currentStoryNum [expr $currentStoryNum + 1];
			}

			#puts "Current maxStoryDriftRatioForFullStrR is $maxStoryDriftRatioForFullStrR"
			#puts "Current minStoryDriftRatioForFullStrR is $minStoryDriftRatioForFullStrR"

		# End of checking for collapse and singularity for this step
		#############################################################################################

	}; # I am not sure why we have two brackets here, but it makes it work!
	}; # I am not sure why we have two brackets here, but it makes it work!

	remove loadPattern $tag
	loadConst -time 0.0

#############################################################################################
# Compute if the EQ is fully converged and output this to the file (output is done below)
	set timeTolerance 2.0;	# We will call it converged as long as it got to within 2.0 seconds of the end of the EQ (in case there is some numerical rounding).
	if {$currentTime > ($maxTime - $timeTolerance)} {
		puts "EQ fully converged"
		set isNonConvR 0
	} else {
		puts "EQ NOT FULLY converged"
		set isNonConvR 1
	}	

# Output the convergence information to the convergence file
	
		set startDir [pwd]
		cd ..
		cd ..
			
		set baseDir [pwd]
		cd $baseDir/Output/$analysisType/EQ_$eqNumber/$SF_DirectoryName
		set runDir [pwd]

		cd $runDir/RunInformation/

		set convFilename [open maxTolUsedOUT.out w]
		puts $convFilename $maxTolUsed
		close $convFilename 

		set tempFile [open isNonConvOUT.out w]
		puts $tempFile $isNonConvR 
		close $tempFile 

		cd $startDir 

# Output a final statement and close the file stream for the convergence log
		# Write a line in the convergence log telling that we are finished and fully converged
		set convFileLine "End of the EQ!"
		puts $convLogFilename $convFileLine 

		close $convLogFilename 


##########################################################################
##########################################################################
############### Older Algorithms #########################################
##########################################################################
##########################################################################

} elseif {$EQsolutionAlgorithm == "CordovaMod"} {
	puts "EQ Solution Algorithm: CordovaMod"
	Don't use this - this has not been updated with the new collapse algo to test for non-conv (see notes on 1-28-05)

	# I changed this to Line Search (it was just Newton) to see if it will make it work (on 6-30-04)
	#     Using the Newton for the first step made the model run for Paul, but it
	#	made my model (when I added the gravity frames) not work at all).  Now 
	#	that I am using NewtonLineSearch from the start, then it's working!!!
	#algorithm Newton
	#algorithm NewtonLineSearch 0.6
	#algorithm NewtonLineSearch 0.8
	#algorithm Newton -initial
	#algorithm KrylovNewton 
	#algorithm KrylovNewton -initial

	system UmfPack

	# Define integrator
	integrator Newmark 0.5  0.25 

	# Define damping
#	rayleigh  $alpha1 $alpha2 0 0; 	# Current stiffness
	rayleigh   $alpha1 0 $alpha2 0;	# Intial stiffness

	# DOF numberer
	numberer RCM

	# Constraint handler
#	constraints Transformation
	constraints $constraintForEQ $constraintArg1EQ $constraintArg2EQ
#	constraints Penalty 1.0e15 1.0e15

	# create the analysis object
	analysis Transient

	# I took this out, so all of the steps are in the analysis loop
	#analyze 1 $dtForAnalysis;	# Paul took one step with Newton when he did his model.

	# switch to NewtonLineSearch - [[ Paul's old comment - "for some reason this does not work 
	# on the very first step, but fine for all succeeding steps"]].

	# Changed here with Frank on 7-6-04 (he reocmended using Newton instead of LineSearch)
	algorithm $iterAlgo

	#Analysis loop - this wil complete when the EQ is over, the structure collapses, or the system goes singular
	while {$ok == 0 && $currentTime < $maxTime && $isCollapsedR == 0 && $isSingularR == 0} {

      	puts "tFinal is $maxTime; and tCurrent is $currentTime"

		# Do step with initial tolerance and input algorithm
		test $testType $testTolerance $testIterations 0
    		set ok [analyze 1 $dtForAnalysis]

		# Keep track of the maximum tolarance used in this step  - for the convPlotFile.  This is later increased it necessary.
		set maxTolUsedInCurrentStep 	[expr $testTolerance]

		#### Change things for convergence ###############
		# If it's not ok, try to decrease dT, but keep the toerance the samecall another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testTolerance]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/10];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, try to decrease dT a bit more, but keep the toerance the samecall another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testTolerance]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/20];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, try to decrease dT a bit more, but keep the toerance the samecall another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testTolerance]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/40];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, try to decrease dT a bit more, but keep the toerance the samecall another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testTolerance]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/80];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance1...call another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testMinTolerance1]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/10];	# This was /20, so maybe change back?
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance1...call another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testMinTolerance1]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/20];	# This was /20, so maybe change back?
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance1...call another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testMinTolerance1]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/40];	# This was /20, so maybe change back?
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance1...call another file for this (just to keep this file clean)
		set currentTolerance 		[expr $testMinTolerance1]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt			[expr $dtForAnalysis/80];	# This was /20, so maybe change back?
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance2...call another file for this (just to keep this file clean)
		set currentTolerance 	[expr $testMinTolerance2]
		set currentNumIterations 	[expr $testLowIter]
		set currentDt		[expr $dtForAnalysis/10];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance3...call another file for this (just to keep this file clean)
		# Decrease dT more
		set currentTolerance 	[expr $testMinTolerance3]
		set currentNumIterations 	[expr $testHighIter]
		set currentDt		[expr $dtForAnalysis/20];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance4...call another file for this (just to keep this file clean)
		# Increase the number of iterations
		set currentTolerance 	[expr $testMinTolerance4]
		set currentNumIterations 	[expr $testHighIter]
		set currentDt		[expr $dtForAnalysis/20];
		source SolutionAlgorithmSubFile.tcl

		# If it's not ok, go to a more relaxed tolerance5...call another file for this (just to keep this file clean)
		set currentTolerance 	[expr $testMinTolerance5]
		set currentNumIterations 	[expr $testHighIter]
		set currentDt		[expr $dtForAnalysis/20];
		source SolutionAlgorithmSubFile.tcl

	# Write a line to the convPlotFile to give the maximumTolerance used at this step (to plot convIndex over time)
		set convPlotFileLine "$currentTime $maxTolUsedInCurrentStep"
		puts $convPlotFilename $convPlotFileLine

	# Update the time
    		set currentTime [getTime]

	# Check if the model went IND (singular matrix).  If it did go IND, then call it collapsed (if useINDAsCollapse == 1), 
	#	otherwise, just let this algorithm continue and it will return an error that the displ is IND and it couldn't
	#	do the calculation below.  Note: Check for all floor nodes, as sometimes the IND or QNAN is not output to all node 
	#	recorders.  NOTICE - this may not work, in the case that you record many nodes and the floor node recorders do not get
	#	updated with QNAN of IND (I am not sure why some recorders get the singularity info, while others do not).  Note that 
	#	sometimes the recorders without the singularity information have huge displacement values in them.
		# First get some node displ responses
			set floor2Displ "[nodeDisp $nodeNumAtFloor2 1]"
			set floor3Displ "[nodeDisp $nodeNumAtFloor3 1]"
			set floor4Displ "[nodeDisp $nodeNumAtFloor4 1]"
			set floor5Displ "[nodeDisp $nodeNumAtFloor5 1]"

			set checkForQNANFlr2 	[string first QNAN $floor2Displ 1]
			set checkForINDFlr2 	[string first IND $floor2Displ 1]
			set checkForQNANFlr3 	[string first QNAN $floor3Displ 1]
			set checkForINDFlr3 	[string first IND $floor3Displ 1]
			set checkForQNANFlr4 	[string first QNAN $floor4Displ 1]
			set checkForINDFlr4 	[string first IND $floor4Displ 1]
			set checkForQNANFlr5 	[string first QNAN $floor5Displ 1]
			set checkForINDFlr5 	[string first IND $floor5Displ 1]

			# Also check if the node resposes are too large, b/c often sigularity is show by huge node displ's, and not just
			#	the error.
			set hugeDispTolForSing	1000000.0;
			# If any of the displ are greater than this (abs value), then call if singular.
			if {($floor2Displ > $hugeDispTolForSing) || ($floor2Displ < -$hugeDispTolForSing) || ($floor3Displ > $hugeDispTolForSing) || ($floor3Displ < -$hugeDispTolForSing) || ($floor4Displ > $hugeDispTolForSing) || ($floor4Displ < -$hugeDispTolForSing) || ($floor5Displ > $hugeDispTolForSing) || ($floor5Displ < -$hugeDispTolForSing)} {
				# Call it singular
				set isSingularR 1;
			}
						
		# Now, if we are using QNAN for collapse, check for "collapse"
		# Check to see if the displacement became QNAN.  If it did, then call it collapsed and stop analysis.
		if {($checkForQNANFlr2 != -1) || ($checkForINDFlr2 != -1) || ($checkForQNANFlr3 != -1) || ($checkForINDFlr3 != -1) || ($checkForQNANFlr4 != -1) || ($checkForINDFlr4 != -1) || ($checkForQNANFlr5 != -1) || ($checkForINDFlr5 != -1)} {
			# Set the flag to show that it became singular
			set isSingularR 1;
			# Now, if we are using QNAN for collapse, then call it collapsed
			if {$useINDAsCollapse == 1} {
			# Check to see if the displacement became QNAN.  If it did, then call it collapsed and stop analysis.
				puts "COLLASPED - It worked - due to QNAN or IND we called it COLLAPSED!"
				# Indeterminate, call it collapsed!
				set isCollapsedR 1
			}
		}

	# If this step is not singular, then try the normal drift check for collapse
		if {$isSingularR == 0} {
			# Determine if the structure is collapsed and if it is then stop the analysis
				set floor2Displ [nodeDisp $nodeNumAtFloor2 1]
				set floor3Displ [nodeDisp $nodeNumAtFloor3 1]
				set floor4Displ [nodeDisp $nodeNumAtFloor4 1]
				set floor5Displ [nodeDisp $nodeNumAtFloor5 1]

				# Compute drift ratios with drift divided by story heights (from the list)
				set story1DriftRatio [expr ($floor2Displ) / [lindex $floorHeightsLIST 1]]
				set story2DriftRatio [expr ($floor3Displ - $floor2Displ) / [lindex $floorHeightsLIST 2]]
				set story3DriftRatio [expr ($floor4Displ - $floor3Displ) / [lindex $floorHeightsLIST 3]]
				set story4DriftRatio [expr ($floor5Displ - $floor4Displ) / [lindex $floorHeightsLIST 4]]

				# Update the max/min drifts for full structure
					if {($story1DriftRatio > $maxStoryDriftRatioForFullStrR)} {
						set maxStoryDriftRatioForFullStr [expr $story1DriftRatio]
					}
					if {($story2DriftRatio > $maxStoryDriftRatioForFullStrR)} {
						set maxStoryDriftRatioForFullStrR [expr $story2DriftRatio]
					}
					if {($story3DriftRatio > $maxStoryDriftRatioForFullStrR)} {
						set maxStoryDriftRatioForFullStrR [expr $story3DriftRatio]
					}
					if {($story4DriftRatio > $maxStoryDriftRatioForFullStrR)} {
						set maxStoryDriftRatioForFullStrR [expr $story4DriftRatio]
					}
					if {($story1DriftRatio < $minStoryDriftRatioForFullStrR)} {
						set minStoryDriftRatioForFullStrR [expr $story1DriftRatio]
					}
					if {($story2DriftRatio < $minStoryDriftRatioForFullStrR)} {
						set minStoryDriftRatioForFullStrR [expr $story2DriftRatio]
					}
					if {($story3DriftRatio < $minStoryDriftRatioForFullStrR)} {
						set minStoryDriftRatioForFullStrR [expr $story3DriftRatio]
					}
					if {($story4DriftRatio < $minStoryDriftRatioForFullStrR)} {
						set minStoryDriftRatioForFullStrR [expr $story4DriftRatio]
					}


		# Checks
#		puts "maxStoryDriftRatioForFullStrR is: $maxStoryDriftRatioForFullStrR"
#		puts "minStoryDriftRatioForFullStrR is: $minStoryDriftRatioForFullStrR"
#
				# Find if it's collapsed - if any floor has too high of a story drift
				if {($maxStoryDriftRatioForFullStrR > $minStoryDriftRatioForCollapse ) || ($minStoryDriftRatioForFullStrR < [expr -($minStoryDriftRatioForCollapse)])} {
					# Changed the flag to make the EQ analysis stop
					set isCollapsedR 1

					# NOT USED...
#					# Now define the variable in the scope of the file that called this function, effectively returning the isCollapsed variable
#					#	to the calling function (Tcl text page 138)
#					uplevel "set isCollapsedForCurrentRun [expr $isCollapsed]"
				}
		}
	}
	remove loadPattern $tag
	loadConst -time 0.0


# Output the convergence information to the convergence file
	
		set startDir [pwd]
		cd ..
		cd ..
			
		set baseDir [pwd]
		cd $baseDir/Output/$analysisType/EQ_$eqNumber/$SF_DirectoryName
		set runDir [pwd]

		cd $runDir/RunInformation/

		set convFilename [open maxTolUsedOUT.out w]
		puts $convFilename $maxTolUsed
		close $convFilename 

		cd $startDir 

# Output a final statement and close the file stream for the convergence log
		# Write a line in the convergence log telling that we are finished and fully converged
		set convFileLine "End of the EQ!"
		puts $convLogFilename $convFileLine 

		close $convLogFilename 

##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################

} else {
	puts "ERROR: No EQ solution algoritm used - check analysis options" 
	ERROR
}

# Output file saying whether or not it used NormDisplIncr
			set startDir [pwd]
			cd ..
			cd ..
			
			set baseDir [pwd]
			cd $baseDir/Output/$analysisType/EQ_$eqNumber/$SF_DirectoryName
			set runDir [pwd]

			file mkdir RunInformation; 
			cd $runDir/RunInformation/

#			set filenameEQ [open usedNormDisplIncrOUT.out w]
#			puts $filenameEQ $usedNormDisplIncr
#			close $filenameEQ 

#			# Output the scaled Sa,comp.  This is from Matlab now (6-29-06).
#			set filename [open saCompScaledOUT.out w]
#			puts $filename $saCompScaled
#			close $filename 
#
#			# Output the scaled Sa,geoMean.  This comes from Matlab now (6-29-06) and not that it is =-1 if we sacled the 
#			#	GMs by Sa,comp (b/c when scaling by Sa,comp we likely did not define both GM components) (6-29-06)
#			set filename [open saGeoMeanScaledOUT.out w]
#			puts $filename $saGeoMeanScaled
#			close $filename 

			# Output the scale factor - This was sent from Matlab and is the factor applied to the component (6-29-06)
			set filename [open scaleFactorAppliedToCompOUT.out w]
			puts $filename $scaleFactorForRunFromMatlab
			close $filename 

			# Output the file format and scaling mthod used
			set filename [open eqFormatForCollapseListOUT.out w]
			puts $filename $eqFormatForCollapseList
			close $filename 

			cd $startDir 

# Return if it collapsed or not - not anymore, as I am using "upvar"
#return $isCollapsed 

}

#####################################################################################################
#####################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

# A procedure for performing section analysis (only does
# 	moment-curvature, but can be easily modified to do any mode
# 	of section reponse.  Note that this procedure uses four node tag
# 	numbers and two element tag numbers (starting at the numbers provided),
# 	so be sure that you input appropriate numbers so that duplications 
# 	don't occur.
#
# Note that this procedure defines load pattern 345-346, but it really isn't important.
#
# This procedure goes out of the model folder into the Output folder and places the files in
#	the output folder for the correct analysis type (correct model).
#
# Curt Haselton
# February 2004
#
# Alter from file by:
# MHS
# October 2000
#
# Arguments;
##	#secTag -- tag identifying section to be analyzed
#	mPhiDirection - indicate positive or negative bending - pos represents
#		the orientation that the section is in in Pauls model now and 
#		negative is for the section inverted.
#	node and element tags - these are input so that the user can do the M-Phi analysis
#		within other analysis and node duplicate tag numbers (thus creating errors).
#	axialLoad -- axial load applied to section (negative is compression)
#	maxK -- maximum curvature reached during analysis
#	numIncr -- number of increments used to reach maxK (default 100)
#
# Sets up a recorder which writes moment-curvature results to file
# "MCurv...section$secTag.out..." - the moment is in column 1, and curvature in column 2

proc MomentCurvatureForPosOrNegMoments {analysisType model secTag axialLoad nodeStartTag eleStartTag locationOfFiber maxK mPhiDirection {numIncr 100000} } {

	puts "Running M-Curv for section $secTag at axialLoad $axialLoad"
		

	# Get into correct folder for output
	cd ..
	cd ..
	set baseDir [pwd];		#Sets baseDir as the current directory
	cd $baseDir/Output/ 
	cd $baseDir/Output/$analysisType; 
	file mkdir MCurvOutput; 
	cd $baseDir/Output/$analysisType/MCurvOutput

	# Set DOF for the loading
	set dof 3

	# Make load patern tag
	set loadPatternOne 345;	#Just a random number
	set loadPatternTwo 346;	#Just a random number

	# Define two nodes at (0,0)
	set nodeOneTag [expr $nodeStartTag]
	set nodeTwoTag [expr $nodeStartTag + 1]
	set eleTag $eleStartTag 
	node $nodeOneTag 0.0 0.0
	node $nodeTwoTag 0.0 0.0

	# Fix all degrees of freedom except axial and bending
	fix $nodeOneTag 1 1 1
	fix $nodeTwoTag 0 1 0

	# Define element
	#                         tag ndI ndJ  secTag
	element zeroLengthSection  $eleTag $nodeOneTag $nodeTwoTag $secTag


	# TEST - try to make a file


	# Create recorder
	recorder Node -file MCurv_Sec_($secTag)_($mPhiDirection)_Axial_($axialLoad).out -time -node $nodeTwoTag -dof $dof disp

	# Do steel recorder and core concrete based on the location input 

		if {$mPhiDirection == "pos"} {
			# Positive bending
			set zSteelFib 0.0;
			set ySteelFib [expr -$locationOfFiber];
			set zCoreConcFib 0.0;
			set yCoreConcFib [expr $locationOfFiber - 1.0];

			# Temp
			set zExtremeConcFib 0.0
			set yExtremeConcFib [expr $locationOfFiber + 2.6];

		} else {
			# Negative bending
			set zSteelFib 0.0;
			set ySteelFib [expr $locationOfFiber];
			set zCoreConcFib 0.0;
			set yCoreConcFib [expr -$locationOfFiber + 1.0];

			# Temp
			set zExtremeConcFib 0.0
			set yExtremeConcFib [expr -$locationOfFiber - 2.6];
		}

	# Do fiber recorders
		recorder Element -file SteelStrain_Sec_($secTag)_($mPhiDirection)_Axial_($axialLoad).out -time -ele $eleTag section fiber $ySteelFib $zSteelFib stressStrain
		recorder Element -file CoreConcStrain_Sec_($secTag)_($mPhiDirection)_Axial_($axialLoad).out -time -ele $eleTag section fiber $yCoreConcFib $zCoreConcFib stressStrain
		# Temp - extreme compression fiber
		recorder Element -file ExtCompFiberStrain_Sec_($secTag)_($mPhiDirection)_Axial_($axialLoad).out -time -ele $eleTag section fiber $yExtremeConcFib $zExtremeConcFib stressStrain


	# Define constant axial load
	pattern Plain $loadPatternOne "Constant" {
		load $nodeTwoTag $axialLoad 0.0 0.0
	}

	# Define analysis parameters
	integrator LoadControl 0 1 0 0
	system SparseGeneral -piv;	# Overkill, but may need the pivoting!
	test NormUnbalance 1.0e-9 10
	numberer Plain
	constraints Plain
	algorithm Newton
	analysis Static

	# Do one analysis for constant axial load
	analyze 1

	# Define the curvature increment based on whether we want the M-Phi for pos or neg curvature
	if {$mPhiDirection == "pos"} {
		set dKStart [expr $maxK/$numIncr]
	} else {
		set dKStart [expr -$maxK/$numIncr]
	}

	# Define reference moment
	set refMoment 1.0	
	pattern Plain $loadPatternTwo "Linear" {
		load $nodeTwoTag 0.0 0.0 $refMoment
	}

	# Did just have "analyze $numIncr" here, I think


	#Analysis loop - used from the EqLoading procedure #################################################
	set currentIncr 1
	set ok 0

	while {$ok == 0 && $currentIncr < $numIncr} {
	
		set dK $dKStart
		# Use displacement control at node for section analysis
		integrator DisplacementControl $nodeTwoTag $dof $dK 1 $dK $dK

   		set ok [analyze 1]
		if {$ok == 0} {
			set currentIncr [expr $currentIncr + 1]
		}	

    		if {$ok != 0} {
			puts "that failed - lets try decreasing the step size by 20 and decreasing tolerance..."
			set dK [expr $dKStart/20]
			integrator DisplacementControl $nodeTwoTag $dof $dK 1 $dK $dK
			test NormUnbalance 1.0e-8 20
			set ok [analyze 1]
			if {$ok == 0} {
				set currentIncr [expr $currentIncr + 1]
			}	
			# Back to initial values
			set dK [expr $dKStart]
			integrator DisplacementControl $nodeTwoTag $dof $dK 1 $dK $dK
			test NormUnbalance 1.0e-9 10
    		}



		# For some reason, it doesn't look like these later loops are ever used - even if it doesn't converge!

    		if {$ok != 0} {
			puts "that failed - lets try decreasing the step size by 20 and using NormDispIncr (with decreased tolerance)..."
			set dK [expr $dKStart/20]
			integrator DisplacementControl $nodeTwoTag $dof $dK 1 $dK $dK
			test NormDispIncr 1.0e-7 20
			set ok [analyze 1]
			if {$ok == 0} {
				set currentIncr [expr $currentIncr + 1]
			}	
			# Back to initial values
			set dK [expr $dKStart]
			integrator DisplacementControl $nodeTwoTag $dof $dK 1 $dK $dK
			test NormUnbalance 1.0e-9 10
    		}
    		if {$ok != 0} {
			puts "that failed - lets try decreasing the step size by 20 and using NormDispIncr (with further decreased tolerance)..."
			set dK [expr $dKStart/40]
			integrator DisplacementControl $nodeTwoTag $dof $dK 1 $dK $dK
			test NormDispIncr 1.0e-6 20
			set ok [analyze 1]
			if {$ok == 0} {
				set currentIncr [expr $currentIncr + 1]
			}	
			# Back to initial values
			set dK [expr $dKStart]
			integrator DisplacementControl $nodeTwoTag $dof $dK 1 $dK $dK
			test NormUnbalance 1.0e-9 10
    		}
    		#if {$ok != 0} {
		#	puts "that failed - lets try decreasing the step size by 30..."
		#	set dK [expr $dKStart/30]
		#	integrator DisplacementControl $nodeTwoTag $dof $dK 1 $dK $dK
		#	set ok [analyze 1]
		#	set currentIncr [expr $currentIncr + 1]
    		#}
	}

#############################################################################3

	# Get back to model folder
	cd ..
	cd ..
	cd ..
	cd $baseDir/Models/$model
	
	# Tell what directory this ended in - for testing
	set tempDir [pwd]
	puts "Current Director is: $tempDir"
}


####################################################################################################
# A procedure for performing section analysis (only does
# 	axial load-strain, but can be easily modified to do any mode
# 	of section reponse.  Note that this procedure uses four node tag
# 	numbers and two element tag numbers (starting at the numbers provided),
# 	so be sure that you input appropriate numbers so that duplications 
# 	don't occur.
#
# Note that this procedure defines load pattern 345-346, but it really isn't important.
#
# This procedure goes out of the model folder into the Output folder and places the files in
#	the output folder for the correct analysis type (correct model).
#
# Curt Haselton
# February 2004
#
# Alter from file by:
# MHS
# October 2000
#
# Arguments
#	secTag -- tag identifying section to be analyzed
#	loadDirection - indicate positive or negative axial load - pos represents
#		tension and negative is compression.
#	node and element tags - these are input so that the user can do the M-Phi analysis
#		within other analysis and node duplicate tag numbers (thus creating errors).
#	maxStrain -- maximum strain reached during analysis
#	numIncr -- number of increments used to reach maxK (default 1000)
#

proc LoadStrainForPosOrNegLoads {analysisType model secTag nodeStartTag eleStartTag maxK loadDirection {numIncr 1000} } {

	puts "Running P-strain for section $secTag"
	
	# Get into correct folder for output
	cd ..
	cd ..
	set baseDir [pwd];		#Sets baseDir as the current directory
	cd $baseDir/Output/ 
	cd $baseDir/Output/$analysisType; 
	file mkdir LoadStrainOutput; 
	cd $baseDir/Output/$analysisType/LoadStrainOutput

	# Set DOF for the loading - axial load is 1
	set dof 1

	# Make load patern tag
	set loadPatternOne 345;	#Just a random number
	set loadPatternTwo 346;	#Just a random number

	# Define two nodes at (0,0)
	set nodeOneTag [expr $nodeStartTag]
	set nodeTwoTag [expr $nodeStartTag + 1]
	set eleTag $eleStartTag 
	node $nodeOneTag 0.0 0.0
	node $nodeTwoTag 0.0 0.0

	# Fix all degrees of freedom except axial 
	fix $nodeOneTag 1 1 1
	fix $nodeTwoTag 0 1 1

	# Define element
	#                         tag ndI ndJ  secTag
	element zeroLengthSection  $eleTag $nodeOneTag $nodeTwoTag $secTag

	# Create recorder
	recorder Node ($analysisType)_Pstrain_Sec_($secTag)_($loadDirection).out disp -time -node $nodeTwoTag -dof $dof 


	# Define analysis parameters
	integrator LoadControl 0 1 0 0
	system SparseGeneral -piv;	# Overkill, but may need the pivoting!
	test NormUnbalance 1.0e-9 10
	numberer Plain
	constraints Plain
	algorithm Newton
	analysis Static

	# Do one analysis for constant axial load
	analyze 1

	# Define the curvature increment based on whether we want the M-Phi for pos or neg curvature
	if {$loadDirection== "pos"} {
		set dKStart [expr $maxK/$numIncr]
	} else {
		set dKStart [expr -$maxK/$numIncr]
	}

	# Define reference load
	set refLoad 1.0	
	pattern Plain $loadPatternTwo "Linear" {
		load $nodeTwoTag $refLoad 0.0 0.0
	}

	# Did just have "analyze $numIncr" here, I think


	#Analysis loop - used from the EqLoading procedure #################################################
	set currentIncr 1
	set ok 0

	while {$ok == 0 && $currentIncr < $numIncr} {
	
		set dK $dKStart
		# Use displacement control at node for section analysis
		integrator DisplacementControl $nodeTwoTag $dof $dK 1 $dK $dK

   		set ok [analyze 1]
		set currentIncr [expr $currentIncr + 1]

    		if {$ok != 0} {
			#puts "that failed - lets try decreasing the step size by 10..."
			set dK [expr $dKStart/10]
			integrator DisplacementControl $nodeTwoTag $dof $dK 1 $dK $dK
			set ok [analyze 1]
			set currentIncr [expr $currentIncr + 1]
    		}
	}

#############################################################################3

	# Get back to model folder
	cd ..
	cd ..
	cd ..
	cd $baseDir/Models/$model
	
	# Tell what directory this ended in - for testing
	set tempDir [pwd]
	puts "Current Director is: $tempDir"
}





####################################################################################################
# Procedure - CreateColumnSection: Defines a procedure which generates a rectangular reinforced concrete section
#  with two outer layers of longitudinal bars and multiple intermediate layers of longitudinal bars, evenly 
# spaced between the outer layers.  Note that the "cover" is both used for both the clear cover and 
# the side cover.  For each intermediate layer, two bars are placed at each of the layers.
#
# Original file written by: Paul Cordova
# Date: 09/2001
#
# Altered by: Curt Haselton 
# Date: 03/2004
# 
#                       y
#                       |
#                       |
#                       |    
#             ---------------------
#             |			    |
#             |  o o o o o o o o  |
#             |   		    |			Ex. numBotBars = 8
#             |  o             o  |			    numTopBars = 8
#  z ---------|                   |  h		    numInterLayers = 3 
#	        |  o 		 o  |
#		  |			    |	
#		  |  o      	 o  |	
#             |      	          |
#             |  o o o o o o o o  |
#             |                   |
#             ---------------------
#                       b
#
# Formal arguments
#    id - tag for the section that is generated by this procedure
#    h - overall height of the section (see above)
#    b - overall width of the section (see above)
#    cover - thickness of the cover patches (from edge to center of bar)
#    coreID - material tag for the core patch
#    coverID - material tag for the cover patches
#    steelID - material tag for the reinforcing steel
#    longBarArea - area of each longitudinal reinforcing bar (same for top/bot bars) 
#    numBotBars - number of tension (bottom) bars
#    numTopBars - number of compression (top) bars
#    interBarArea - area of each intermediate reinforcing bar 
#    numInterLayers - number of intermediate layers of bars (2 bars per intermediate layer)
#    nfCoreY - number of fibers in the core patch in the y direction
#    nfCoreZ - number of fibers in the core patch in the z direction
#    nfCoverY - number of fibers in the cover patches with long sides in the y direction
#    nfCoverZ - number of fibers in the cover patches with long sides in the z direction
#
# Notes
#    The thickness of cover concrete is constant on all sides of the core.
#    The number of bars is the same on any given side of the section.
#    The reinforcing bars are all the same size.
#    The number of fibers in the short direction of the cover patches is set to 1.
# 
proc CreateColumnSection {id h b cover coreID coverID steelID longBarArea numBotBars numTopBars interBarArea numInterLayers nfCoreY nfCoreZ nfCoverY nfCoverZ} {

   # Do outputs for testing
#   puts "id is $id"
#   puts "h is $h"
#   puts "cover is $cover"
#   puts "coreid is $coreID"
#   puts "coverID is $coverID"
#   puts "steelID is $steelID"
#   puts "longBarArea is $longBarArea"
#   puts "numBotBars is $numBotBars"
#   puts "numTopBars is $numTopBars"
#   puts "interBarArea is $interBarArea"
#   puts "numInterLayers is $numInterLayers"
#   puts "nfCoreY is $nfCoreY"
#   puts "nfCoreZ is $nfCoreZ"
#   puts "nfCoverY is $nfCoverY"
#   puts "nfCoverZ is $nfCoverZ"

   # Set number of intermediate bars per layer
   set numInterBarsPerLayer 2

   # The distance from the section z-axis to the edge of the cover concrete
   # in the positive y direction
   set coverY [expr $h/2.0]

   # The distance from the section y-axis to the edge of the cover concrete
   # in the positive z direction
   set coverZ [expr $b/2.0]

   # The negative values of the two above
   set ncoverY [expr -$coverY]
   set ncoverZ [expr -$coverZ]

   # Determine the corresponding values from the respective axes to the
   # edge of the core concrete
   set coreY [expr $coverY-$cover]
   set coreZ [expr $coverZ-$cover]
   set ncoreY [expr -$coreY]
   set ncoreZ [expr -$coreZ]

   # Define the fiber section
   section fiberSec $id {

	# Define the core patch - CHECKED and good (9-13-04)
	patch quadr $coreID $nfCoreZ $nfCoreY $ncoreY $coreZ $ncoreY $ncoreZ $coreY $ncoreZ $coreY $coreZ
      
	# Define the four cover patches  - I changed this on 9-13-04 - The top and bottom covers were discretized wrong.
	# Side cover
	patch quadr $coverID $nfCoverZ $nfCoreY $ncoverY $coverZ $ncoreY $coreZ $coreY $coreZ $coverY $coverZ
	patch quadr $coverID $nfCoverZ $nfCoreY $ncoreY $ncoreZ $ncoverY $ncoverZ $coverY $ncoverZ $coreY $ncoreZ
	# Top and bottom cover
	patch quadr $coverID $nfCoverZ $nfCoverY $ncoverY $coverZ $ncoverY $ncoverZ $ncoreY $ncoreZ $ncoreY $coreZ
	patch quadr $coverID $nfCoverZ $nfCoverY $coreY $coreZ $coreY $ncoreZ $coverY $ncoverZ $coverY $coverZ

	# Define the steel layers
	# Top layer
	layer straight $steelID $numTopBars $longBarArea $coreY $ncoreZ $coreY $coreZ; 	
	# Bottom layer
	layer straight $steelID $numBotBars $longBarArea $ncoreY $ncoreZ $ncoreY $coreZ; 

	# Do intermediate layers if they exist
#	puts "Check: Number of intermeddiate layers is $numInterLayers"
   	if {$numInterLayers == 0} {
		# Don't make any intermediate layers of bars
   	} else {
		# Compute spacing of intermediate layers (equally spaced)
		set interSpacing [expr (($h - 2*$cover)/[expr $numInterLayers + 1])]
		#puts "Inter spacing is $interSpacing"
		
		# Make intermediate layers
		set layerDepth [expr ($coreY - $interSpacing)]
		for {set layerNum 1} {$layerNum < [expr $numInterLayers + 1]} {incr layerNum 1} {
			layer straight $steelID $numInterBarsPerLayer $interBarArea $layerDepth $coreZ $layerDepth $ncoreZ; 
			#puts "Intermediate layer placed at $layerDepth"
			set layerDepth [expr $layerDepth - $interSpacing]
		}
   	}	
   }
}
####################################################################################################
# Procedure - CreateBeamWithSlabSection: Defines a procedure which generates a rectangular reinforced concrete section
#  with two outer layers of longitudinal bars and multiple intermediate layers of longitudinal bars, evenly 
# spaced between the outer layers.  Note that the "cover" is both used for both the clear cover and 
# the side cover.  For each intermediate layer, two bars are placed at each of the layers.
# The slab is added to the -z side of the section, but it really doesn't matter b/c of the PSRP 
# assumption.  Slab steel is placed at the top and the bottom of the slab section, a distance
# "slab cover" from the top and bototm of the section.  A slab bar is only placed if the slab width
# is >= the slab bar spacing.
#
# If slab width or slab thinkness is < 0, then no slab is created.  
# If slabBarSpacing is > slabWidth, then no slab stell is defined. 
#
# Original file written by: Paul Cordova
# Date: 09/2001
#
# Altered by: Curt Haselton 
# Date: 03/2004
# 
#                       y
#                       |
#                       |	    |<-----------slabWidth----------->|
#                       |    
#             -------------------------------------------------------
#             |			    |o             o             o   
#             |  o o o o o o o o  |
#             |   		    |o_____________o_____________o_____	
#             |  o             o  |			    
#  z ---------|                   |  h		    
#	        |  o 		 o  |				
#		  |			    |			Ex. 	numBotBars = 8
#		  |  o      	 o  |				numTopBars = 8
#             |      	          |				numInterLayers = 3 
#             |  o o o o o o o o  |
#             |                   |
#             ---------------------
#                       b
#
# Formal arguments
#    id - tag for the section that is generated by this procedure
#    h - overall height of the section (see above)
#    b - overall width of the section (see above)
#    cover - thickness of the cover patches (from edge to center of bar) for the beam
#    coreID - material tag for the core patch
#    coverID - material tag for the cover patches
#    slabID - material tag for the slab patches
#    steelID - material tag for the reinforcing steel
#    longBarArea - area of each longitudinal reinforcing bar (same for top/bot bars) 
#    numBotBars - number of tension (bottom) bars
#    numTopBars - number of compression (top) bars
#    interBarArea - area of each intermediate reinforcing bar 
#    numInterLayers - number of intermediate layers of bars (2 bars per intermediate layer)
#    slabWidth - slab width in addition to the beam width
#    slabThick - slab thickness
#    slabBarArea -  area of each slab steel reinforcing bar (same for top/bot bars) 
#    slabBarSpacing - spacing between sets of slab bars - BE SURE THIS IS NOT ZERO BECAUSE OF DIVISION!
#    slabBarCover - cover to center of slab bars - used for both top and bottom bar layers
#    nfCoreY - number of fibers in the core patch in the y direction
#    nfCoreZ - number of fibers in the core patch in the z direction
#    nfCoverY - number of fibers in the cover patches with long sides in the y direction
#    nfCoverZ - number of fibers in the cover patches with long sides in the z direction
#    nfSlabY - number of fibers in the slab with long sides in the y direction
#    nfSlabZ - number of fibers in the slab with long sides in the z direction
#
# Notes
#    The thickness of cover concrete is constant on all sides of the core.
#    The number of bars is the same on any given side of the section.
#    The reinforcing bars are all the same size.
#    The number of fibers in the short direction of the cover patches is set to 1.
# 
proc CreateBeamWithSlabSection {id h b cover coreID coverID slabID steelID longBarArea numBotBars numTopBars interBarArea numInterLayers slabWidth slabThick slabBarArea slabBarSpacing slabBarCover nfCoreY nfCoreZ nfCoverY nfCoverZ nfSlabY nfSlabZ} {

   # Do outputs for testing
#   puts "id is $id"
#   puts "h is $h"
#   puts "cover is $cover"
#   puts "coreid is $coreID"
#   puts "coverID is $coverID"
#   puts "steelID is $steelID"
#   puts "longBarArea is $longBarArea"
#   puts "numBotBars is $numBotBars"
#   puts "numTopBars is $numTopBars"
#   puts "interBarArea is $interBarArea"
#   puts "numInterLayers is $numInterLayers"
#   puts "nfCoreY is $nfCoreY"
#   puts "nfCoreZ is $nfCoreZ"
#   puts "nfCoverY is $nfCoverY"
#   puts "nfCoverZ is $nfCoverZ"

   # Set number of intermediate bars per layer
   set numInterBarsPerLayer 2

   # The distance from the section z-axis to the edge of the cover concrete
   # in the positive y direction
   set coverY [expr $h/2.0]

   # The distance from the section y-axis to the edge of the cover concrete
   # in the positive z direction
   set coverZ [expr $b/2.0]

   # The negative values of the two above
   set ncoverY [expr -$coverY]
   set ncoverZ [expr -$coverZ]

   # Determine the corresponding values from the respective axes to the
   # edge of the core concrete
   set coreY [expr $coverY-$cover]
   set coreZ [expr $coverZ-$cover]
   set ncoreY [expr -$coreY]
   set ncoreZ [expr -$coreZ]

   # Define the fiber section
   section fiberSec $id {

	# Define the core patch - OK - checked on 9-13-04
	patch quadr $coreID $nfCoreZ $nfCoreY $ncoreY $coreZ $ncoreY $ncoreZ $coreY $ncoreZ $coreY $coreZ
      
	# Define the four cover patches  - I changed this on 9-13-04 - The top and bottom covers were discretized wrong.
	# Side cover
	patch quadr $coverID $nfCoverZ $nfCoreY $ncoverY $coverZ $ncoreY $coreZ $coreY $coreZ $coverY $coverZ
	patch quadr $coverID $nfCoverZ $nfCoreY $ncoreY $ncoreZ $ncoverY $ncoverZ $coverY $ncoverZ $coreY $ncoreZ
	# Top and bottom cover
	patch quadr $coverID $nfCoverZ $nfCoverY $ncoverY $coverZ $ncoverY $ncoverZ $ncoreY $ncoreZ $ncoreY $coreZ
	patch quadr $coverID $nfCoverZ $nfCoverY $coreY $coreZ $coreY $ncoreZ $coverY $ncoverZ $coverY $coverZ
	# Left cover patch
#	patch quadr $coverID $nfCoverZ $nfCoverY $ncoreY $coverZ $ncoreY $coreZ $coreY $coreZ $coreY $coverZ
#	# Right cover patch
#	patch quadr $coverID $nfCoverZ $nfCoverY $ncoreY $ncoreZ $ncoreY $ncoverZ $coreY $ncoverZ $coreY $ncoreZ
#	# Bottom cover patch
#	patch quadr $coverID $nfCoverZ $nfCoverY $ncoverY $coverZ $ncoverY $ncoverZ $ncoreY $ncoverZ $ncoreY $coverZ
#	# Top cover patch
#	patch quadr $coverID $nfCoverZ $nfCoverY $coverY $coverZ $coverY $ncoverZ $coverY $ncoverZ $coverY $coverZ
	
	# Define the steel layers
	# Top layer
	layer straight $steelID $numTopBars $longBarArea $coreY $ncoreZ $coreY $coreZ; 	
	# Bottom layer
	layer straight $steelID $numBotBars $longBarArea $ncoreY $ncoreZ $ncoreY $coreZ; 

	# Do intermediate layers if they exist
   	if {$numInterLayers == 0} {
		# Don't make any intermediate layers of bars
   	} else {
		# Compute spacing of intermediate layers (equally spaced)
		set interSpacing [expr (($h - 2*$cover)/[expr $numInterLayers + 1])]
		#puts "Inter spacing is $interSpacing"
		
		# Make intermediate layers
		set layerDepth [expr ($coreY - $interSpacing)]
		for {set layerNum 1} {$layerNum < [expr $numInterLayers + 1]} {incr layerNum 1} {
			layer straight $steelID $numInterBarsPerLayer $interBarArea $layerDepth $coreZ $layerDepth $ncoreZ; 
			#puts "Intermediate layer placed at $layerDepth"
			set layerDepth [expr $layerDepth - $interSpacing]
		}
   	}

	# Define the patch for the slab, if there is a slab -    
	set rightSlabZ [expr $ncoverZ - $slabWidth]
	set botSlabY [expr $coverY - $slabThick]	

	if {$slabWidth > 0 && $slabThick > 0} {
		#     					   bot/left   	   bot/right             top/right           top/left          	              
		# Careful here - need to define quadr in CCW around the patch!!!
		patch quadr $slabID $nfSlabZ $nfSlabY $botSlabY $ncoverZ $botSlabY $rightSlabZ $coverY $rightSlabZ $coverY $ncoverZ  
		#puts "Slab patch defined!"
		#puts "Slab ID is: $slabID"
	}

 	# Define steel layers for the slab
     		# Compute top and bottom slab layer heights
     		set topSlabBarY [expr $coverY - $slabBarCover]
     		set botSlabBarY [expr $botSlabY + $slabBarCover]

		# Compute number of bars to put in layers in slab - this should round to an integer result
     		set numSlabBarsInLayer [expr int($slabWidth/$slabBarSpacing)]

     		# Place layers, if there are bars to place - top then bottom
		if {$numSlabBarsInLayer > 0} {
     			layer straight $steelID $numSlabBarsInLayer $slabBarArea $topSlabBarY $ncoverZ $topSlabBarY $rightSlabZ
     			layer straight $steelID $numSlabBarsInLayer $slabBarArea $botSlabBarY $ncoverZ $botSlabBarY $rightSlabZ
			#puts "Slab layers defined with $numSlabBarsInLayer bars per layer!"
		}
   }
}
####################################################################################################


############### Start of Ratational Spring Proceedure ##############################
# rotSpring2D.tcl
# Procedure which creates a rotational spring for a planar problem
#
# SETS A MULTIPOINT CONSTRAINT ON THE TRANSLATIONAL DEGREES OF FREEDOM,
# SO DO NOT USE THIS PROCEDURE IF THERE ARE TRANSLATIONAL ZEROLENGTH
# ELEMENTS ALSO BEING USED BETWEEN THESE TWO NODES
#
# Written: MHS
# Date: Jan 2000
#
# Formal arguments
#	eleID - unique element ID for this zero length rotational spring
#	nodeR - node ID which will be retained by the multi-point constraint
#	nodeC - node ID which will be constrained by the multi-point constraint
#	matID - material ID which represents the moment-rotation relationship
#		for the spring

proc rotSpring2D {eleID nodeR nodeC matID} {
	# Create the zero length element
	element zeroLength $eleID $nodeR $nodeC -mat $matID -dir 6

	# Constrain the translational DOF with a multi-point constraint
	#          retained constrained DOF_1 DOF_2 ... DOF_n
	equalDOF    $nodeR     $nodeC     1     2  	

}

############### End of Rotational Spring Proceedure ##############################


###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################

# READSDMFILE2.TCL
# ------------------------------------------------------------------------------------------------------------read gm input format
#
# Written: MHS
# Date: July 2000
#
# Altered: Curt Haselton of Stanford University
# Date: September 2004
#
# A procedure which parses a ground motion record from the PEER
# 	strong motion database by finding dt in the record header, then
# 	echoing data values to the output file.
# An important note is how the variables dt and numPoints are handled: these
#	variables are effectively passed into this function by reference, so the
#	variables that you pass in for dt and numPoints will be altered by this 
#	function.  To use this, in your main script set dt and numPoints both to 0.0,
#	then call ReadSMDFile2 and pass in all of the variables including dt and numPoints.
#	After you call ReadSMDFile2, the dt and numPoints variables in your main script
#	will be redifined by this function, having the correct values from the PEER SMD
#	file (e.g. something like dt = 0.005 and numPoints = 8000).  If you have any questions,
#	you can e-mail me at haselton@stanford.edu.
#
# Formal arguments
#	inFilename -- file which contains PEER strong motion record
#	outFilename -- file to be written in format G3 can read
#	dt -- time step determined from file header (this is updated by this function - PBR)
#	numPoints -- the number of points in the EQ record, from the file header (this is updated by this function - PBR)
#
# Assumptions
#	The header in the PEER record is, e.g., formatted as follows:
#	 PACIFIC ENGINEERING AND ANALYSIS STRONG-MOTION DATA
#	  IMPERIAL VALLEY 10/15/79 2319, EL CENTRO ARRAY 6, 230                           
#	  ACCELERATION TIME HISTORY IN UNITS OF G                                         
#	  NPTS=  3930, DT= .00500 SEC

proc ReadSMDFile2 {inFilename outFilename dt numPoints} {
	
	# Pass dt and numPoints by reference.  This passes the values back to the calling function.
	upvar dt DT;	
	upvar numPoints NUMPOINTS;	# I added this line to also give me the number of points in the EQ record (from the header)

	# Open the input file and catch the error if it can't be read
	if [catch {open $inFilename r} inFileID] {
		puts stderr "Cannot open $inFilename for reading"
	} else {
		# Open output file for writing
		set outFileID [open $outFilename w]

		# Flag indicating dt is found and that ground motion
		# values should be read -- ASSUMES dt and numPoints is on last line
		# of header!!!
		set dtFlag 0
		set numPointsFlag 0

		# Look at each line in the file
		foreach line [split [read $inFileID] \n] {

			if {[llength $line] == 0} {
				# Blank line --> do nothing
				continue
			} elseif {$dtFlag == 1} {
				# If it gets here, then it is done with the header part of the file.
				# Echo ground motion values to output file, but only put ONE value per line, instead of FIVE per line, so opensees can read it!
				foreach accelDataPoint [split $line] {
					if {$accelDataPoint != " " && $accelDataPoint != ""} {
						# If it's not a blank value, then put into outFile
						puts $outFileID $accelDataPoint 
					}
				}

			} else {
				# Search header lines for dt
				foreach word [split $line] {
					# Read in the time step
					if {$dtFlag == 1} {
						# This loop will catch the dt AFTER the numPoint is found!
						set DT $word
						break;	# Break if we got the time step because now we are done with the header.
					} elseif {$numPointsFlag == 1} {
						if {$word != "" && $word != " "} {
							# This loop will catch the numPoints.
							set NUMPOINTS $word
							# Reset flag so we don't come back into this statement
							set numPointsFlag 0
						}
					}
					# Find the desired token and set the flag, so that in the next loop we will catch the information that we need.
					if {[string match $word "DT="] == 1} {
						set dtFlag 1
					}
					if {[string match $word "NPTS="] == 1} {
						puts "Found NPTS=!!!"
						set numPointsFlag 1
					} 
				}
			}
		}

		# Close the output file
		close $outFileID

		# Close the input file
		close $inFileID
	}

	# Sometimes the NUMPOINTS value has a comma at the end of the token, so if the last part is a comma, then remove it from the
	#	varaible, so that it is a integer.
		# Find if there is a comma at the end of the string, NUMPOINTS...
		set searchString ","
		set isAComma [string last $searchString $NUMPOINTS];	
		# Remove the comma if it is there (this assumes that the comma is always at the end of the string)
		if {$isAComma != -1} {
			# Remove the last character of the string, which should be the comma
			set NUMPOINTS [string trimright $NUMPOINTS ,]
		}

#	puts "At end of the proc..."
#	puts "numPointsFlag is $numPointsFlag"
#	puts "DT is $DT"
#	puts "NUMPOINTS is $NUMPOINTS"
}

########################################################################################## 


###########################################################################################################

# ReadSMDFile_PEER-NGA-Format.TCL
# ------------------------------------------------------------------------------------------------------------read gm input format
#
# Procedure originally written by: MHS
# Date: July 2000
#
# File altered to work for PEER-NGA format: Curt Haselton of Stanford University
# Date: August 2005
#
# A procedure which parses a ground motion record from the PEER
# 	strong motion database by finding dt in the record header, then
# 	echoing data values to the output file.
# An important note is how the variables dt and numPoints are handled: these
#	variables are effectively passed into this function by reference, so the
#	variables that you pass in for dt and numPoints will be altered by this 
#	function.  To use this, in your main script set dt and numPoints both to 0.0,
#	then call ReadSMDFile2 and pass in all of the variables including dt and numPoints.
#	After you call ReadSMDFile2, the dt and numPoints variables in your main script
#	will be redifined by this function, having the correct values from the PEER SMD
#	file (e.g. something like dt = 0.005 and numPoints = 8000).  If you have any questions,
#	you can e-mail me at haselton@stanford.edu.
#
# Formal arguments
#	inFilename -- file which contains PEER strong motion record
#	outFilename -- file to be written in format G3 can read
#	dt -- time step determined from file header (this is updated by this function - PBR)
#	numPoints -- the number of points in the EQ record, from the file header (this is updated by this function - PBR)
#
# Assumptions - This is the PEER-NGA format; NOT the PEER format!
#	- The header in the PEER record is, e.g., formatted as follows:
#		PEER NGA STRONG MOTION DATABASE RECORD
#		CORINTH EQ, 02/24/81, M6.7, CORINTH, TRANSVERSE                                 
#		ACCELERATION TIME HISTORY IN UNITS OF G
#		4094    0.0100    NPTS, DT
#		(after this come the accelTH values)
#
#	- Note: This procedure processes the record by looking for "G" and then after it finds it it saves the next value as the 
#		number of time steps in the record and the value after that as the time step.
#
#	- In the PEER-NGA format, there are six accel values per line, but this processor makes no asusmptions about the number of 
#		acceleration values per line (it just devides them up until the line is gone). 
#
#

proc ReadSMDFile_PEER-NGA-Format {inFilename outFilename dt numPoints} {
	
	# Pass dt and numPoints by reference.  This passes the values back to the calling function.
	upvar dt DT;	
	upvar numPoints NUMPOINTS;	# I added this line to also give me the number of points in the EQ record (from the header)

	# Open the input file and catch the error if it can't be read
	if [catch {open $inFilename r} inFileID] {
		puts stderr "Cannot open $inFilename for reading"
	} else {
		# Open output file for writing
		set outFileID [open $outFilename w]

		# Flag indicating dt is found and that ground motion
		# values should be read -- ASSUMES dt and numPoints is on last line
		# of header!!!
		set dtFlag 0
		set numPointsFlag 0
		set foundNumPointsFlag 0
		#set GFlag 0


		# Look at each line in the file
		foreach line [split [read $inFileID] \n] {

			if {[llength $line] == 0} {
				# Blank line --> do nothing
				continue
			} elseif {$dtFlag == 1} {
				# If it gets here, then it is done with the header part of the file.
				# Echo ground motion values to output file, but only put ONE value per line, instead of SIX per line, so opensees can read it!
				foreach accelDataPoint [split $line] {
					if {$accelDataPoint != " " && $accelDataPoint != ""} {
						# If it's not a blank value, then put into outFile
						puts $outFileID $accelDataPoint 
					}
				}

			} else {
				# Search header lines for G
				foreach word [split $line] {
					# Read in the time step
					if {$dtFlag == 1} {
						# This loop will catch the dt and numPoints after "G" is found!
						# If the line is not blank a space, it will save the dT and break, otherewise it will 
						#	continue looping and looking for dt until there	is not a blank space
						if {$word != "" && $word != " "} {					
							set DT $word
							break;	# Break if we got the time step because now we are done with the header.
						} else {
							# Do not do anything and wait until the next loop to record dt (when we find something other than a blank space)
						}
					} elseif {$numPointsFlag == 1} {
						if {$word != "" && $word != " "} {
							# This loop will catch the numPoints.
							set NUMPOINTS $word
							# Reset flag so we don't come back into this statement
							set foundNumPointsFlag 1
							set numPointsFlag 0
						}
					}
					# If we found the number of points (flag = 1), then set the dT flag, so we read the dt value on the next loop
					if {($foundNumPointsFlag == 1)} {
						set dtFlag 1
					}
					# If we find the "G" set the flag so we know that the NPTS is the next token
					if {[string match $word "G"] == 1} {
						puts "Found NPTS,!!!"
						set numPointsFlag 1
					} 
				}
			}
		}

		# Close the output file
		close $outFileID

		# Close the input file
		close $inFileID
	}

	# Sometimes the NUMPOINTS value has a comma at the end of the token, so if the last part is a comma, then remove it from the
	#	varaible, so that it is a integer.
		# Find if there is a comma at the end of the string, NUMPOINTS...
		set searchString ","
		set isAComma [string last $searchString $NUMPOINTS];	
		# Remove the comma if it is there (this assumes that the comma is always at the end of the string)
		if {$isAComma != -1} {
			# Remove the last character of the string, which should be the comma
			set NUMPOINTS [string trimright $NUMPOINTS ,]
		}

#	puts "At end of the proc..."
#	puts "numPointsFlag is $numPointsFlag"
#	puts "DT is $DT"
#	puts "NUMPOINTS is $NUMPOINTS"
#	puts "dtFlag is $dtFlag"
#	puts "numPointsFlag is $numPointsFlag"

}

########################################################################################## 


###########################################################################################################

# ReadSMDFile_PEER-NGA_Rotated-Format.TCL
# ------------------------------------------------------------------------------------------------------------read gm input format
#
# Procedure originally written by: MHS
# Date: July 2000
#
# File altered to work for PEER-NGA-Rotated format: Curt Haselton of Stanford University
# Date: May 2006
#
# This is the same as "ReadSMDFile_PEER-NGA-Format.TCL", except it has been slightly modified to read the format of the 
#	PEER-NGA data file for a record that has been rotated to be Fault-Normal/Fault-Parallel.  The slight modification is 
#	that I look for "g" instead of "G" to find where the dT and numPoints is in the header of the file. 
#
# Formal arguments
#	inFilename -- file which contains PEER strong motion record
#	outFilename -- file to be written in format G3 can read
#	dt -- time step determined from file header (this is updated by this function - PBR)
#	numPoints -- the number of points in the EQ record, from the file header (this is updated by this function - PBR)
#
# Assumptions - This is the PEER-NGA format; NOT the PEER format!
#	- The header in the PEER record is, e.g., formatted as follows:
#  		PEER NGA Rotated Accelerogram (v1.1)
#  		H1 for rotation: LOMA PRIETA 10/18/89 00:05, BRAN, 090                                           
#  		H2 for rotation: LOMA PRIETA 10/18/89 00:05, BRAN, 000                                           
#  		rotation angle - clockwise    38.0
#  		FP component, azimuth =   128.0
#  		Acceleration in g
#      	5001   0.00500 NPTS, DT
#		(after this come the accelTH values)
#
#	- Note: This procedure processes the record by looking for "g" and then after it finds it it saves the next value as the 
#		number of time steps in the record and the value after that as the time step.
#
#	- In the PEER-NGA format, there are six accel values per line, but this processor makes no asusmptions about the number of 
#		acceleration values per line (it just devides them up until the line is gone). 
#

proc ReadSMDFile_PEER-NGA-Rotated-Format {inFilename outFilename dt numPoints} {
	
	# Pass dt and numPoints by reference.  This passes the values back to the calling function.
	upvar dt DT;	
	upvar numPoints NUMPOINTS;	# I added this line to also give me the number of points in the EQ record (from the header)

	# Open the input file and catch the error if it can't be read
	if [catch {open $inFilename r} inFileID] {
		puts stderr "Cannot open $inFilename for reading"
	} else {
		# Open output file for writing
		set outFileID [open $outFilename w]

		# Flag indicating dt is found and that ground motion
		# values should be read -- ASSUMES dt and numPoints is on last line
		# of header!!!
		set dtFlag 0
		set numPointsFlag 0
		set foundNumPointsFlag 0
		#set GFlag 0


		# Look at each line in the file
		foreach line [split [read $inFileID] \n] {

			if {[llength $line] == 0} {
				# Blank line --> do nothing
				continue
			} elseif {$dtFlag == 1} {
				# If it gets here, then it is done with the header part of the file.
				# Echo ground motion values to output file, but only put ONE value per line, instead of SIX per line, so opensees can read it!
				foreach accelDataPoint [split $line] {
					if {$accelDataPoint != " " && $accelDataPoint != ""} {
						# If it's not a blank value, then put into outFile
						puts $outFileID $accelDataPoint 
					}
				}

			} else {
				# Search header lines for G
				foreach word [split $line] {
					# Read in the time step
					if {$dtFlag == 1} {
						# This loop will catch the dt and numPoints after "G" is found!
						# If the line is not blank a space, it will save the dT and break, otherewise it will 
						#	continue looping and looking for dt until there	is not a blank space
						if {$word != "" && $word != " "} {					
							set DT $word
							break;	# Break if we got the time step because now we are done with the header.
						} else {
							# Do not do anything and wait until the next loop to record dt (when we find something other than a blank space)
						}
					} elseif {$numPointsFlag == 1} {
						if {$word != "" && $word != " "} {
							# This loop will catch the numPoints.
							set NUMPOINTS $word
							# Reset flag so we don't come back into this statement
							set foundNumPointsFlag 1
							set numPointsFlag 0
						}
					}
					# If we found the number of points (flag = 1), then set the dT flag, so we read the dt value on the next loop
					if {($foundNumPointsFlag == 1)} {
						set dtFlag 1
					}
					# If we find the "g" set the flag so we know that the NPTS is the next token
					if {[string match $word "g"] == 1} {
						puts "Found NPTS,!!!"
						set numPointsFlag 1
					} 
				}
			}
		}

		# Close the output file
		close $outFileID

		# Close the input file
		close $inFileID
	}

	# Sometimes the NUMPOINTS value has a comma at the end of the token, so if the last part is a comma, then remove it from the
	#	varaible, so that it is a integer.
		# Find if there is a comma at the end of the string, NUMPOINTS...
		set searchString ","
		set isAComma [string last $searchString $NUMPOINTS];	
		# Remove the comma if it is there (this assumes that the comma is always at the end of the string)
		if {$isAComma != -1} {
			# Remove the last character of the string, which should be the comma
			set NUMPOINTS [string trimright $NUMPOINTS ,]
		}

#	puts "At end of the proc..."
#	puts "numPointsFlag is $numPointsFlag"
#	puts "DT is $DT"
#	puts "NUMPOINTS is $NUMPOINTS"
#	puts "dtFlag is $dtFlag"
#	puts "numPointsFlag is $numPointsFlag"

}
########################################################################################## 

############### Start of Bi-Translational Spring Procedure ##############################
# biDirectionalSpring2D.tcl
# Procedure which creates two translational springs for a planar problem.  Two materials are input and these materials are
#	used to describe the force-displacement response between the nodes, for DOF's 1 and 2.  matID1 describes DOF 1 and similarly
#	for matID2.  The third DOF is left free (no restraint).
#
# Original version written: MHS
# Date: Jan 2000
#
# Extended by: Curt Haselton
# Date: September 2004
#
# Formal arguments
#	eleID - unique element ID for this zero length rotational spring
#	nodeR - node ID which will be retained by the multi-point constraint
#	nodeC - node ID which will be constrained by the multi-point constraint
#	matID1 - material ID which represents force-displcement response in the X direction
#	matID2 - material ID which represents force-displcement response in the Y direction

proc biDirectionalSpring2D {eleID nodeR nodeC matID1 matID2} {

	# Do error check to be sure that either dof 1 or 2 are specified
#	if {$dof != 1 && $dof != 2} {
#		puts "ERROR - The translationalSpring2D can only be used for DOF 1 or 2!!!"
#		ERROR - this line stops analysis
#	}

	# Create the zero length element
	element zeroLength $eleID $nodeR $nodeC -mat $matID1 $matID2 -dir 1 2

#	# Constrain the other DOF's with a multi-point constraint
#	    	#      	retained 	constrained 	DOF_1 DOF_2 ... DOF_n
#		equalDOF    $nodeR     	$nodeC     		3	


}

############### End of biDirectional Spring Proceedure ##############################


############### Start of bilinearBondMaterial Procedure ##############################
# Procedure: bilinearPinchingMaterial.tcl
#
# Procedure which creates a bilinear pinched bond-slip material (Hysteretic) according to my notes around 12-14-04 for 
#	the new designs and models.  This material can later be used in the RotSpring2D to make the spring.  This was made for bond-slip
#	but is now also bing used for the shear in the element sections.
#
# Written by: Curt Haselton
# Date: December 2004
#
# Formal arguments
#	matID - unique element ID for this zero length rotational spring
#	stressMax - yield moment of section (but this material does not actually yield, it just kinks)
#	strainMax - yield rotation of section (but this material does not actually yield, it just kinks)
#	stiffRatio - ratio of first a seconds stiffness (proposed to use 4.0)
#	kinkStressRatio - ratio of Mmax (use 0.4)
#	pinchX - pinching ratio of Hysteretic model (use 0.5 for 25% displ. pinching)
#	pinchY - pinching ratio of Hysteretic model (use 0.25 for 25% force pinching)

proc bilinearPinchingMaterial {matID stressMax strainMax stiffRatio kinkStressRatio pinchX pinchY} {

# Compute the points on the Hysteretic backbone - see notes on 12-14-04
	# Kink point
	set s1p 	[expr $stressMax * $kinkStressRatio ]
	set e1p 	[expr $strainMax * (($kinkStressRatio / $stiffRatio) / (($kinkStressRatio / $stiffRatio) + (1 - $kinkStressRatio)))]

	# Yield point
	set s2p 	[expr $stressMax]
	set e2p 	[expr $strainMax]

	# Point past yield (just bilinear)
	set stiffness 	[expr ($s2p - $s1p) / ($e2p - $e1p)];	# Stiffness of seconds slope
	set strainOffset 	[expr $strainMax];					# Just convenient to use
	set s3p		[expr $s2p + ($stiffness * $strainOffset)]
	set e3p		[expr $e2p + $strainOffset]

set s1n [expr -$s1p]
set s2n [expr -$s2p]
set s3n [expr -$s3p]

set e1n [expr -$e1p]
set e2n [expr -$e2p]
set e3n [expr -$e3p]

# Create the material
#					matTag	s1p  e1p  s2p  e2p  s3p  e3p		s1n  e1n  s2n  e2n  s3n  e3n 		pinchX 	pinchY	damage1	damage2 	<beta>
uniaxialMaterial Hysteretic 	$matID	$s1p $e1p $s2p $e2p $s3p $e3p		$s1n $e1n $s2n $e2n $s3n $e3n		$pinchX  	$pinchY	0		0		0	

}

############### End of bilinearBondMaterial Proceedure ##############################

############### Start of Translational Shear Spring Proceedure ##############################
# transSpring2D.tcl
# Procedure which creates a translational spring for a planar problem
#
# SETS A MULTIPOINT CONSTRAINT ON THE TRANSLATIONAL DEGREES OF FREEDOM,
# SO DO NOT USE THIS PROCEDURE IF THERE ARE TRANSLATIONAL ZEROLENGTH
# ELEMENTS ALSO BEING USED BETWEEN THESE TWO NODES
#
# Written: MHS
# Date: Jan 2000
#
# Altered to be flexible spring to capture ground motion displacements by Abbie Liel
# Date: Dec 2005
#
# Formal arguments
#	eleID - unique element ID for this zero length rotational spring
#	nodeR - node ID which will be retained by the multi-point constraint
#	nodeC - node ID which will be constrained by the multi-point constraint
#	matID - material ID which represents the moment-rotation relationship
#		for the spring

proc transSpring2D {eleID nodeR nodeC matID} {
	# Create the zero length element
	element zeroLength $eleID $nodeR $nodeC -mat $matID -dir 1

	# Constrain the translational DOF with a multi-point constraint
	#          retained constrained DOF_1 DOF_2 ... DOF_n
	equalDOF	$nodeR	$nodeC	2	3  	

}
############### End of Translational Spring Proceedure ##############################

############### Start of Ibarra Material Procedure ########################################################
# Procedure: CreateIbarraMaterial.tcl
#
# Procedure which creates a uniaxial material created by Ibarra, Median, Krawinkler, and company.  This
#	material is called "Clough" in the Opensees implementation.  This material is typically used to 
#	represent moment-plasticRotation of an element plastic hinge, but the material is general to any response
#	type (e.g. you could also use it for stress-strain of force-displacement responses).
#
# You can use stiffFactor1 and stiffFactor2 when using this material as a plastic hinge (where you want the initial
#	stiffness very rigid.  If you assume double curvature in the element, you can use stiffFactor1=11 and stiffFactor2=1.1 
#	then multiply your elastic element stiffness by stiffFactor2=1.1.  If you use these factors and use this material
#	procedure, then the total element stiffnesses (or the combined elastic element and plastic hinge) will be correct.  
#	If you want to just a standard material model with no odd stiffness adjustement factors, 
#	just use stiffFactor1=stiffFactor2=1.0. However, note that you are still inputing EIeff and this procedure uses the 
#	eleLength and 6EI/L to compute the rotational stiffness of the spring.
#	
# The input lambda value is used for the basic strength and cappign strength deterioration, 2*lambda is used for the unloading 
#	stiffness deterioration, and lambda = 0 is used for the accelerated reloading stiffness deterioration.
#
# Written by: Curt B. Haselton, Stanford University
# Date: June 8, 2006
#
# Formal arguments
#	matID - unique material ID number
#	EIeff - effective section stiffness
#	myPos - positive flexural strength
#	myNeg - SIGNED negative flexural strength
#	mcOverMy - ratio of (Mc moment at capping in the positive direction) to myPos
#	thetaCapPos - the total rotation at capping, but when you use stiffFactor=11, then this is approximatley the plastic rotation at capping
#	thetaCapNeg - SIGNED rotation; same as thetaCapPos, but in the negative direction
#	thetaPC - rotation from capping to zero strength (in both positive and negative directions)
#	lambda - normalized cyclic deterioration capacity; this should be the value normalized by the initial stiffness of the element (i.e. the stiffFactors 
#		are applied within this procedure)
#	c - cyclic deterioration exponent
#	resStrRatio - residual strengthl this should be as a ratio of the non-degraded strength, but in some of the code, this is actually 
#		a ratio of the reduced strength after accounting for cycluc damage
#	stiffFactor1, stiffFactor2 - stiffness factors - use 11 and 1.1 for a stiff plastic hinge, use 1.0 and 1.0 for no stiffness adjustment.
#	eleLength - length of element, used for stiffness calculation (to get rotational stiffness from EIeff); this assumed that you supply the 
#		full element length and that the element is in double curvature.
#
################################################################################################################

proc CreateIbarraMaterial { matTag EIeff myPos myNeg mcOverMy thetaCapPos thetaCapNeg thetaPC lambda c resStrRatio stiffFactor1 stiffFactor2 eleLength } {

	# Do needed calculations
		set elstk 	$EIeff;	# Initial elastic stiffness
		set alphaHardUnscaled 	[expr ((($myPos*$mcOverMy)-$myPos)/($thetaCapPos)) / $elstk];
		set alphaHardScaled	[expr $alphaHardUnscaled * ((-$stiffFactor2 * $alphaHardUnscaled ) / ($alphaHardUnscaled * ($alphaHardUnscaled - $stiffFactor2)))];	# This altered the stiffness to account for the elastic element stiffness - see hand notes on 1-6-05
		set alphaCapUnscaled	[expr ((-$myPos*$mcOverMy)/($thetaPC)) / $elstk];
		set alphaCapScaled	[expr $alphaCapUnscaled * ((-$stiffFactor2 * $alphaCapUnscaled) / ($alphaCapUnscaled * ($alphaCapUnscaled - $stiffFactor2)))];	# This altered the stiffness to account for the elastic element stiffness - see hand notes on 1-6-05
		set lambdaA 0; 							# No accelerated stiffness deterioration
		set lambdaS [expr $lambda * $stiffFactor1];		# Strength
		set lambdaK 0;							# No unloading stiffness deterioration because there is a bug in this portion of the model
		#set lambdaK [expr 2.0 * $lambda * $stiffFactor1];	# Unloading stiffness (2.0 is per Ibarra chapter 3)
		set lambdaD [expr $lambda * $stiffFactor1];		# Capping strength

	# Create the material model
		uniaxialMaterial Clough  $matTag $elstk $myPos $myNeg $alphaHardScaled $resStrRatio $alphaCapScaled	$thetaCapPos $thetaCapNeg $lambdaS $lambdaK $lambdaA $lambdaD  $c $c $c $c
	
	# Checks
		#puts "matTag is $matTag"
		#puts "elstk is $elstk"
		#puts "myPos is $myPos"
		#puts "myNeg is $myNeg"
		#puts "mcOverMy is $mcOverMy"
		#puts "alphaHardScaled is $alphaHardScaled"
		#puts "resStrRatio is $resStrRatio"
		#puts "alphaCapScaled is $alphaCapScaled"
		#puts "thetaCapPos is $thetaCapPos"
		#puts "thetaCapNeg is $thetaCapNeg"
		#puts "thetaPC is $thetaPC"
		#puts "lambdaS is $lambdaS"
		#puts "lambdaK is $lambdaK"
		#puts "lambdaA is $lambdaA"
		#puts "lambdaD is $lambdaD"
		#puts "c is $c"
}

#proc CreateIbarraMaterial_temp {matID EIeff myPos myNeg mcOverMy thetaCapPlPos thetaCapPlNeg thetaPC lambda c stiffFactor1 stiffFactor2 } {
#
#	# Create the material model
#		# Compute the elastic stiffness, accounting for the factor that may be used to make the pre-yield rotation small (usually 11 is used)
#		set elstk 	[expr $stiffFactor1 * ((6 * $EIeff) / $eleLength)]
#
#
#				# Hardening and softening stiffnesses - use the positive direction to compute it.
#				set alphaHardUnscaled 	[expr ((($myPos*$mcOverMy)-$myPos)/($thetaCapPlPos)) / $elstk];
#				set alphaCapUnscaled	[expr ((-$myPos*$mcOverMy)/($thetaPc)) / $elstk];
#
#
#
#
#
#				set alphaHardToUse	[expr $stiffFactor1 * $alphaUnscaled * ((-$stiffFactor2 * $alphaUnscaled) / ($alphaUnscaled * ($alphaUnscaled - $stiffFactor2)))];
#				set alphaCapToUse		[expr $stiffFactor1 * $alphaCapUnscaled * ((-$stiffFactor2 * $alphaCapUnscaled) / ($alphaCapUnscaled * ($alphaCapUnscaled - $stiffFactor2)))]			
#				# Deterioration parameters
#				set lambdaS [expr $lambda * $stiffFactor1];		# Strength
#				set lambdaD [expr $lambda * $stiffFactor1];		# Capping strength
#				set lambdaK [expr 2.0 * $lambda * $stiffFactor1];	# Unloading stiffness (2.0 is per Ibarra chapter 3)
#				set lambdaA [expr 0.0];							# Accelerated reloading stiffness (set to zero because calibration showed this is appropriate for RC columns)
#
#
#				#                       tag     		    E      fy+    fy-    alphaS          Resfac  alphaC         thetaCapPlPos  thetaCapPlNeg  lambdaS  lambdaK  lambdaA  lambdaD  cs  ck  ca  cd
#				uniaxialMaterial Clough  $CS1Pof218kUniaxialT $elstk $myPos $myNeg $alphaHardToUse $resFac $alphaCapToUse $thetaCapPlPos $thetaCapPlNeg $lambdaS $lambdaK $lambdaA $lambdaD $c $c $c $c
#
#}
#

############### End of Ibarra Material Procedure ########################################################

# puts "All functions and procedures defined"
