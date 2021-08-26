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
		set elstk 	[expr $stiffFactor1 * ((6 * $EIeff) / $eleLength)];	# Initial elastic stiffness
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


