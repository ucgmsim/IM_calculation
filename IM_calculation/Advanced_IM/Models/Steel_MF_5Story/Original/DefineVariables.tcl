#----------------------------------------------------------------------------------#
# DefineVariables
#	This file can hold any defined quantity, but NEEDS to hold all of the variables 
#	that will be varied in the analysis (for sensitivity studies).  
#	They should be in this file because the program calls this file to define all 
#	of the variables, then alters the parameter that should be altered for the 
#	sensitivity study.
#
#	Note that constants can be defined anywhere in the file structure, but values that 
#	could be varied need to be defined in this module.
#
# Units: kips, in, sec
#
# This file developed by: Curt Haselton of Stanford University
# Updated: 28 June 2005
# Date: 17 Sept 2003
#
# Other files used in developing this model:
#		none
#----------------------------------------------------------------------------------#

# Notes:
#	- Section dimensions must be in this file because the material file uses the dimensions for SPP calculations
#	- The order in this file is pretty random


# Stiffness factor that is used now (6-8-06).  Apply a stiffness factor to make the model by for Kstf40 rather than Kyld; make this 1.0 for the archtype model and just use for sensitivity!!
	set eleStiffF 2.2;

# Make a list of elements for which to define damping (using the region command later).  Only add LE elements 
#	so we don't get spurious damping forces in NL elements (Medina's thesis, Appendix A.5l 1994 Bernal paper).
# 	This damping is defined in the RunEQ... or RunEQCol..tcl files (and/or the associated solution algortihm file).
	set listOfElementsForRayleighDamping {1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36};

# Set factors on the dead load and live loads
	set DLF		1.05;		# Dead load factor - based on work by Ellingwood - this is set up to also scale the mass (i.e. DL is used to compute masses)
	set LLF		1.0;		# Live load factor

# Factors on element strength
	#set hystHingeStrengthF		1.0;
	set hingeStrF	1.0;		# Hinge strength for all elements (this scales yield strength and hardened strength proportionally)
	set hingeStrBmF	1.0;		# Hinge yield strength factors for Bm and col (in addition to the hingeStrF) - Hyst hinge model and PinchingDamage model
	set hingeStrColF	1.0;

## Factors on element stiffness
	set hingeYStfBmF	1.0;	# WARNING - This MAY BE controlled by other variables now
	set hingeYStfColF	1.0;	# WARNING - This MAY BE controlled by other variables now

# Set factors for the lumped plasticity model
	set cappingRotationF 	1.0;	# This only affects the pinchDmg model
	set yieldedPHStiffF	1.0;	# WARNING - Leave this as 1.0!  This only affects the clough model, but this is 
					#	altered in the DefineMultiple...tcl file based on the value of topLevelYldStiffF.
	set phLengthF 1.0;	# This applies to elements of the main frame and the columns of the gravity frame.

# Design factors - for all models
set SCWBDesF 	1.0;		# (values: 0.74, 1.0, 1.26) This is a factor on the SCWB factor, but is not the actaul facor of around 1.2!  When SCWBDesF = 1.0, 
					#	then the SCWB factor is at the mean value. This only scales the column strength.
set bmDesStrF 	1.0;		# (values: 0.65, 1.0, 1.35) This is a factor on the beam design strength. When bmDesStrF = 1.0, then the beam strength
					#	ratio (overstrength) is at the mean value.  This value scales the beam and column strengths
					#	proportionally, so that the SCWB ratio is maintained.

# SIMPLIFY THIS FOR ARCHETYPES!
# Joint shear information - please see hand notes on 9-1-04, and updates on 9-15-04
	set crackingShearStrain 		0.0002;	# Shear strain at cracking.  From 4th WCEE reference - see hand notes on 9-15-04.
	set yieldShearStrain			0.004;	# This is the shear strain at yield of the joint.  From 4th WCEE reference - see hand notes on 9-15-04.
	set ratioOfShearStrengthAtCracking	0.26;		# This is the ratio of the shear strength at the cracking point (full strength, i.e. the ACI strength multiplied by the factor.
								#	From 4th WCEE reference - see hand notes on 9-15-04.
	set jntCrackStrF				1.00;		# Factor for joint cracking strength
	set jointStrengthF			1.40;		# Factor on joint shear strength - this is multiplied by the nominal joint strength, from ACI 318-02 calc. (with no phi reductions)
								# 	TuseLEJointShearPanel useLEJointShearPanel useLEJointShearPanel his is from notes on 9-9-04, from test data.  This generally agrees (a bit lower) with 4th WCEE reference
	set allowJointShearFailure		YES;		# This will allow the joint to fail (at the strength in the DefineMaterials.tlc file, multiplied by the jointStrF) and use the postFailureSlopeRatio after failure
	#set allowJointShearFailure		NO;		# This will not allow the joint to fail, but will just let it crack at teh correnct ratio of the strength, then continue with the cracked slope forever.
	set jointShearCapSlope			-0.07;	# See hand notes on 9-1/9-04.  Slope after failure, as ratio of the secnt slope to the yield point, if failure is allowed
	set jointShearCapSlopeF			1.00;		# Factor on the jointShearCapSlope
	set jointResidStrF			0.00;		# Joint residual strength (ratio of capcacity) - only really matters if you soften joint, which I am not doing
	set jointShearForcePinchingRatioPos	0.25; #1.00 - for conv.		# Pinching ratio for joint (pos value) - ratio of max historic stress, for pos only (Hand notes on 9-1-04, Arash, thesis - pg. 113)
	set jointShearForcePinchingRatioNeg	0.25;	#1.00 - for conv.		# Pinching ratio for joint (pos value) - ratio of max historic stress, for neg only (Hand notes on 9-1-04, Arash, thesis - pg. 113)
	set jointShearDispPinchingRatio	0.25;	#0.00 - for conv.		# Pinching ratio for displ (pos value) (ratio of max historic displ), both pos/neg (Hand notes on 9-1-04, Arash, thesis - pg. 113)

# Only for Hysteretic model (shouldn't be used for Buffalo model)
	set hystJointShearPinchX		0.50;	# This is used for the pinching of the joint shear this values is for pinching at 25% of max historic force (see single ele test folder on 8-12-04)
	set hystJointShearPinchY		0.25;	# This is used for the pinching of the joint shear this values is for pinching at 25% of max historic force (see single ele test folder on 8-12-04)

# Column bases 
	set colBaseRotStiffF	1.0;	# Factor on the stifness of the elastic springs at the bases of the columns of the lateral and gravity frame.
					#	steel for that section (values: 0.79, 1.00, 1.21) for the fiber model and scales strength for clough model.

# Input the conctraint handler to use.  If Penalty is used, then input arguments.
	set constraintForEQ 	Transformation
	set constraintArg1EQ	"";	# Only needed for penalty method, so put "" is other method is used
	set constraintArg2EQ	"";	# Only needed for penalty method, so put "" is other method is used
	#set constraintForEQ 	Penalty
	#set constraintArg1EQ	1.0e15;	# Only needed for penalty method, so put "" is other method is used
	#set constraintArg2EQ	1.0e15;	# Only needed for penalty method, so put "" is other method is used

	#set constraintForPO 	Transformation
	#set constraintArg1PO	"";	# Only needed for penalty method, so put "" is other method is used
	#set constraintArg2PO	"";	# Only needed for penalty method, so put "" is other method is used
	set constraintForPO 	Penalty
	set constraintArg1PO	1.0e15;	# Only needed for penalty method, so put "" is other method is used
	set constraintArg2PO	1.0e15;	# Only needed for penalty method, so put "" is other method is used

# Decide if largeDispl should be used for the joint elements
	set lrgDsp 1;	# Causes a bit more QNAN problems, but better to use.
	#set lrgDsp 0

# Geometric transformations
	# Transformation - for the elements of the lateral frame and the leaning column.
	# Do one for all elements
		set strGeomTransf LinearWithPDelta
		#set strGeoTransf Corotational

	# Transformation for gravity frame - gravity frame beams and columns
		#set gravFrameTranf $strGeoTransf;
		set gravFrameGeomTranf Linear;	# Make the leaning columns take care of all of the P-Delta
		#set gravFrameTranf LinearWithPDelta;

# Define loading information
# REMOVE THIS FOR ARCHETYPE MODEL AFTER SIMPLFYING DL AND MASS STUFF
	set concWtDens		[expr 150.0*$pcf];	#This is used for beam and column DL and mass
	set DistFloorDL 		[expr 110.0*$psf];	#This is for slab and MEP weight
	set DistRoofDL 		[expr 118.0*$psf];	#This is for slab and MEP weight, roof stuff
	set DistFloorLL 		[expr 12.0*$psf]; 	#This is from Ellington reference - arbitrary point in time LL
	set DistRoofLL		[expr 5.0*$psf];	# This is same ratio as Ellington's reference
	set tribWidthExtFrame 	[expr 15.0*$ft]
	set claddingWt		[expr 20.0*$psf]
	set ratioOfLLInMass 0.25; 	# Define what ratio of LL to include in the mass

# Strut material for leaning column
	set E_strut [expr 1.00 * 29000.0*$ksi]
	# set A_strut 1e9
	set A_strut 500
	# set I_strut 1e9
	set I_strut 50000

# Elastic test material stiffness - used for joint hinges not connected to anything, etc.
	set E_elasticTestMaterial 	[expr 1700.0 * 29000.0*$ksi];	# This is made to match the initial bond-slip M-Rot spring stifness for BS1

## Beam dimensions and reinforcing (h = height, b = width)
# REMOVE THIS FOR ARCHETYPE MODEL AFTER SIMPLFYING DL AND MASS STUFF
#	# Dimensions
	set hBS1	42.0
	set hBS2	36.0
	set hBS3	32.0
	set hBS4	32.0

	set bBS1	24.0
	set bBS2	24.0
	set bBS3	24.0
	set bBS4	24.0

	set slabThick 8.0;

# Column dimensions and reinforcing
# REMOVE THIS FOR ARCHETYPE MODEL AFTER SIMPLFYING DL AND MASS STUFF
	# Dimensions
	set hCS1	30.0
	set hCS2	40.0
	set hCS3	33.0
	set hCS4	34.0
	set hCS5	30.0
	set hCS6	28.0
	set hCSGrav	18.0;	# Interior gravity columns

	set bCS1	30.0
	set bCS2	30.0
	set bCS3	30.0
	set bCS4	30.0
	set bCS5	30.0
	set bCS6	24.0
	set bCSGrav	18.0;	# Interior gravity columns

# Define building geometry
	set bayWidth 		[expr 30.0 * $ft]
	set storyOneHeight 	[expr 14.0 * $ft]
	set typicalStoryHeight 	[expr 13.0 * $ft]

	set floorTwoHeight	[expr $storyOneHeight]
	set floorThreeHeight	[expr $storyOneHeight + 1.0*$typicalStoryHeight]
	set floorFourHeight	[expr $storyOneHeight + 2.0*$typicalStoryHeight]
	set floorFiveHeight	[expr $storyOneHeight + 3.0*$typicalStoryHeight]

# Splice locations
# REMOVE THIS FOR ARCHETYPES!!!
	set storyTwoSpliceHeight	[expr $storyOneHeight + 0.5 * ($typicalStoryHeight - $hBS2)]
	set storyThreeSpliceHeight	[expr $storyOneHeight + $typicalStoryHeight + 0.5 * ($typicalStoryHeight - $hBS3)]
	set beamLength			[expr $bayWidth - $hCS4];	# For hinge stiffness calculations (accuracy not too critical)	
	set colLength			[expr $typicalStoryHeight - $hBS2];	# For hinge stiffness calculations (accuracy not too critical)	

# Primary lateral frame column base rotational stifness
	#set lateralColBaseK 12800000;	# This is from hand notes dated 7-23-04/9-16-04

# Define the section iformation for the connectors between the main and gravity frames - note that this models the shear
# 	flexibility of the slab, but I made it stiff, so that there isn't much flexibility.
	set Econnectors	[expr 1.00 * 29000.0*$ksi]
	set Aconnectors 	1600.0;	# Don't let the nodes translate with respect to each other - calibrated to be approx. the stifness of a column (same EA/L)

# Update this for the ARCHETYPES - Automate this!
	set dampRat		0.025;	# Changed to 2.5%
	set dampRatF	1.0;		# Factor on the damping ratio.

	# Define the periods to use for the Rayleigh damping calculations
	set periodForRayleighDamping_1 1.659;	# Mode 1 period - NEEDS to be UPDATED
	set periodForRayleighDamping_2 0.241;	# Mode 3 period - NEEDS to be UPDATED

# Remove for ARCHETYPES
# Define the offset used for the leaning column - pretty arbitrary
	set leaningColOffset [expr 5*$ft]

# These are definitions that are used for the elastic elements axial stifnesses - just reasonale numbers for the stiffness material to be well conditioned
	set EOfUnity	1.0;		# This will be used when EI is used for I
	set Acol		1320.0;	# Just based on conpressive transformed area for CS3 (kind of avergae for now)
	set ABm		1025.0;	# Just based on conpressive transformed area for BS2 (kind of avergae for now)
	set Econcr 4030;	# From ACI calcs, for 5 ksi concrete
	set EAcol 		[expr $Econcr * $Acol]
	set EABm 		[expr $Econcr * $ABm]

# Define variables for the Clough hinges (with damage) (udpated on 6-8-06)
	set resStrRatio 	0.01;	# This is the residual strength ratio; this muct be non-zero or else we see a bug in the unloading/reloading stiffnesses	!!!
	set c		1.0;	# Exponent for deterioration
	set lambda	[expr $eleStiffF * 120.0]; 	# This is for most of the lambda values (calculated in the materials file) - 
												#	lambdaS = lambdaFactor * lambdaAll
												#	lambdaD = lambdaFactor * lambdaAll
												#	lambdaK = 0 (the factor of 2 is per Ibarra chapter 3)
	set lambdaFactor	1.0;	# This factor is applied to lambdaS, lambdaD, and lambdaK
	#set lambdaA		0;	# This makes it not use any accel. stiffness deterioration, other than what comes from the 
					#	strength deterioration modes.

# Define stiffness factors for the plastic hinges and elastic elements.  stiffFactor1 primarily stiffness the PHs and stiffFactor2 stiffens the elastic elements. (6-8-06)
	set stiffFactor1 11.0
	set stiffFactor2 1.1

# Set stiffness of translational spring (see Abbie's notes dated 12-9-05)
	set groundDisplMatK 10000;

# Set a zero rotational stiffness term - this is used when defining the leaning column elements (I SHOULD JUST REMOVE THIS and PUT "0.0" in the element definitions)
	set IOfZero 0.0;

# Add a dummy value for the period for scaling defined by Matlab when running collapse anlayses.  This was added in attempt to avoid
#	an error when trying to output this value if Matlab was not used to drive collapse analyses (i.e. if we were doing stripe analysis 
#	not using Matlab as the driver).
	set periodUsedForScalingGroundMotionsFromMatlab -1;

# Define small masses for convergence.  These are applied by VB to virtually every DOF. 7-12-06
# This is what I used in DesA_Buffalo_v.9noGFrm when you consider the factor of 10 I used.
	set smallMass1 0.03235;	# Translational x
	set smallMass2 0.03235;	# Translational y
	set smallMass3 4.313;	# Rotational

# Define parameters for 1967 frame joint model.  This uses Liel document (File1) in the 1967 arch folder and on e-mail in July 2006 (CBH 8-3-06)
set alphaS_Joint 					0.03;
set Resfac 						0.05;
set jointShearCapSlope 				-0.07;
set jointForcePinchRat				0.25;	# Pos and neg same.
set jointDispPinchRat 				0.25;
set lambdaJ						44.3;
set cJ						1.0;
set jointStrF					1.0;

# Define a few materials not defined by the model code from VBA
	# Define the elastic material used for the SDOF to record ground displacement
	set groundDisplT 88888
	uniaxialMaterial Elastic $groundDisplT $groundDisplMatK

	# Stiff elastic strut material (used as rigid links)
	set strutMatT	599; 	# To agree with leaning column numbering
	uniaxialMaterial Elastic $strutMatT $E_strut

	# Make an elastic material for testing while building the model (this is also used for the joint bond-slip springs that are not connected to anything)
	set elastJointMatT	49999; 	# To agree with joint numbering
	uniaxialMaterial Elastic $elastJointMatT $E_elasticTestMaterial

# Define a node needed by the ground displacement recorder (not defined by VBA code), fix one of 
#	the nodes, define the spring to use, then apply the mass.
	# if {($findgroundDispl == 1)} {
		# node 88001 [expr 0.0] [expr 0.0]
		# node 88002 [expr 0.0] [expr 0.0]
		# fix 88001 1 1 1
		# transSpring2D 88888 88001 88002 $groundDisplT 
		# set largeGDMass 2533030; #units are kips-s^2/in
 		# mass 88002 $largeGDMass 0 0 
	# }

# Set up the geometeric transformations here, so they won't need to be made by the VBA code
	# Geometric transformation - for everything but the gravity frame
	set primaryGeomTransT 1
	geomTransf $strGeomTransf $primaryGeomTransT 
	# Geometric transformation - for the gravity frame
	set gravFrameGeomTranfT 2
	geomTransf $gravFrameGeomTranf $gravFrameGeomTranfT 

	
	
	
# puts "Variables defined"


##############################################################################################################################
#          										  Define Material Prperties										             #
##############################################################################################################################


# define material properties
	set Es 29000.0;														# Steel's Young's modulus
	set Ec 3605.0;														# Concrete's Young's modulus
	set fc 4.0;														    # Steel's Yield Strength
	set fy 50.0;														# Steel's Yield Strength
	set v 0.3;															# Poisson's Ratio for Steel
    set Krigid 1e9;														# Stiffness for rigid joint

#Define residual strength ratio for Clough Beam-Column Elements
	set resStrRatioBeamColumn .01;
 # puts "material properties defined"

##############################################################################################################################
#          										   Define Material and Section Tags									 	     #
##############################################################################################################################


	set BeamHingeMatM1 200001;													# Material tag for beams on the moment frames
	set BeamHingeMatM2 200002;													# Material tag for beams on the moment frames
	set BeamHingeMatM3 200003;													# Material tag for beams on the moment frames
 	set ColHingeMatM1 300001;													# Material tag for columns on the moment frames
 	set ColHingeMatM2 300002;													# Material tag for columns on the moment frames
 	set ColHingeMatM3 300003;													# Material tag for columns on the moment frames
 	set ColHingeMatM4 300004;													# Material tag for columns on the moment frames
	set PanelZoneM1 400001;														# Material tag for panel zones in the moment frames
	set PanelZoneM2 400002;														# Material tag for panel zones in the gravity frames
	set PanelZoneM3 400003;														# Material tag for panel zones in the moment frames
	set PanelZoneM4 400004;														# Material tag for panel zones in the gravity frames
	set PanelZoneM5 400005;														# Material tag for panel zones in the moment frames
	set PanelZoneM6 400006;														# Material tag for panel zones in the gravity frames
	set rigidMat 600001;														# Material tag for rigid linear compenents
	set pinnedMat 600002;														# Material tag for pinned linear compenents
	

	# puts "material tags defined" 


