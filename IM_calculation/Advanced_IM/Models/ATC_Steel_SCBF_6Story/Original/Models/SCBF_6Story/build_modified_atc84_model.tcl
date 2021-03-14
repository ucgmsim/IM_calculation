# --------------------------------------------------------------------------------------------------


###################################################################################################
#          Set Up & Source Definition
###################################################################################################
model BasicBuilder -ndm 2 -ndf 3;	# Define the model builder, ndm = #dimension, ndf = #dofs
source [file join [file dirname [info script]] rotSpring2DModIKModel.tcl];

###################################################################################################
#          Define Building Geometry, Nodes, and Constraints
###################################################################################################
# define structure-geometry parameters
set NStories 6;						# number of stories
set NBays 1;						# number of frame bays (excludes bay for P-delta column)
set WBay      [expr 30.0*12.0];		# bay width in inches
set HStory1   [expr 15.0*12.0];		# 1st story height in inches
set HStoryTyp [expr 15.0*12.0];		# story height of other stories in inches
set HBuilding [expr $HStory1 + ($NStories-1)*$HStoryTyp];	# height of building

# calculate locations of beam/column joints:
set Pier1  0.0;		# leftmost column line
set Pier2  [expr $Pier1 + $WBay];
set Pier3  [expr $Pier2 + $WBay];	# P-delta column line
set Floor1 0.0;		# ground floor
set Floor2 [expr $Floor1 + $HStory1];
set Floor3 [expr $Floor2 + $HStoryTyp];

set Floor4 [expr $Floor3 + $HStoryTyp];
set Floor5 [expr $Floor4 + $HStoryTyp];
set Floor6 [expr $Floor5 + $HStoryTyp];
set Floor7 [expr $Floor6 + $HStoryTyp];

# calculate joint offset distance for beam plastic hinges
set phlat23 [expr 0.0];		# lateral dist from beam-col joint to loc of hinge on Floor 2

# calculate nodal masses -- lump floor masses at frame nodes
set g 386.089

set Floor2Weight 986.0;		# weight of Floor 2 in kips
set Floor2FrameD [expr {$Floor2Weight/12.0}]
set Floor2LeanD [expr {$Floor2Weight*11.0/12.0}]

set Floor3Weight 986.0;		# weight of Floor 3 in kips
set Floor3FrameD [expr {$Floor3Weight/12.0}]
set Floor3LeanD [expr {$Floor3Weight*11.0/12.0}]


set Floor4Weight 986.0;		# weight of Floor 4 in kips
set Floor4FrameD [expr {$Floor4Weight/12.0}]
set Floor4LeanD [expr {$Floor4Weight*11.0/12.0}]

set Floor5Weight 986.0;		# weight of Floor 5 in kips
set Floor5FrameD [expr {$Floor5Weight/12.0}]
set Floor5LeanD [expr {$Floor5Weight*11.0/12.0}]

set Floor6Weight 986.0;		# weight of Floor 6 in kips
set Floor6FrameD [expr {$Floor6Weight/12.0}]
set Floor6LeanD [expr {$Floor6Weight*11.0/12.0}]

set Floor7Weight 757.0;		# weight of Floor 7 in kips
set Floor7FrameD [expr {$Floor7Weight/12.0}]
set Floor7LeanD [expr {$Floor7Weight*11.0/12.0}]

set Floor2LL 540.0
set Floor2FrameL [expr {$Floor2LL/12.0}]
set Floor2LeanL [expr {$Floor2LL*11.0/12.0}]

set Floor3LL 540.0
set Floor3FrameL [expr {$Floor3LL/12.0}]
set Floor3LeanL [expr {$Floor3LL*11.0/12.0}]

set Floor4LL 540.0
set Floor4FrameL [expr {$Floor4LL/12.0}]
set Floor4LeanL [expr {$Floor4LL*11.0/12.0}]

set Floor5LL 540.0
set Floor5FrameL [expr {$Floor5LL/12.0}]
set Floor5LeanL [expr {$Floor5LL*11.0/12.0}]

set Floor6LL 540.0
set Floor6FrameL [expr {$Floor6LL/12.0}]
set Floor6LeanL [expr {$Floor6LL*11.0/12.0}]

set Floor7LL 216.0
set Floor7FrameL [expr {$Floor7LL/12.0}]
set Floor7LeanL [expr {$Floor7LL*11.0/12.0}]



#set Negligible $NodalMass6

# define nodes and assign masses to beam-column intersections of frame
# command:  node nodeID xcoord ycoord -mass mass_dof1 mass_dof2 mass_dof3
# nodeID convention:  "xy" where x = Pier # and y = Floor #
node 11 $Pier1 $Floor1;
node 21 $Pier2 $Floor1;
node 12 $Pier1 $Floor2 
node 22 $Pier2 $Floor2 
node 13 $Pier1 $Floor3 
node 23 $Pier2 $Floor3 

node 14 $Pier1 $Floor4 
node 24 $Pier2 $Floor4 
node 15 $Pier1 $Floor5 
node 25 $Pier2 $Floor5 
node 16 $Pier1 $Floor6 
node 26 $Pier2 $Floor6 
node 17 $Pier1 $Floor7 
node 27 $Pier2 $Floor7 

node 42 [expr $Pier2/2] $Floor2;    #Extra Node to connect Braces at Center
node 44 [expr $Pier2/2] $Floor4;    #Extra Node to connect Braces at Center
node 46 [expr $Pier2/2] $Floor6;    #Extra Node to connect Braces at Center


# define extra nodes for plastic hinge rotational springs
# nodeID convention:  "xya" where x = Pier #, y = Floor #, a = location relative to beam-column joint
# "a" convention: 2 = left; 3 = right;
# "a" convention: 6 = below; 7 = above;
# column hinges at bottom of Story 1 (base)
node 117 $Pier1 $Floor1;
node 217 $Pier2 $Floor1;
node 317 $Pier3 $Floor1;
# column hinges at top of Story 1
node 126 $Pier1 $Floor2;
node 226 $Pier2 $Floor2;
node 326 $Pier3 $Floor2;	# zero-stiffness spring will be used on p-delta column
# column hinges at bottom of Story 2
node 127 $Pier1 $Floor2;
node 227 $Pier2 $Floor2;
node 327 $Pier3 $Floor2;	# zero-stiffness spring will be used on p-delta column
# column hinges at top of Story 2
node 136 $Pier1 $Floor3;
node 236 $Pier2 $Floor3;
node 336 $Pier3 $Floor3;	# zero-stiffness spring will be used on p-delta column

# column hinges at bottom of Story 3
node 137 $Pier1 $Floor3;
node 237 $Pier2 $Floor3;
node 337 $Pier3 $Floor3;	# zero-stiffness spring will be used on p-delta column
# column hinges at top of Story 3
node 146 $Pier1 $Floor4;
node 246 $Pier2 $Floor4;
node 346 $Pier3 $Floor4;	# zero-stiffness spring will be used on p-delta column
# column hinges at bottom of Story 4
node 147 $Pier1 $Floor4;
node 247 $Pier2 $Floor4;
node 347 $Pier3 $Floor4;	# zero-stiffness spring will be used on p-delta column
# column hinges at top of Story 4
node 156 $Pier1 $Floor5;
node 256 $Pier2 $Floor5;
node 356 $Pier3 $Floor5;	# zero-stiffness spring will be used on p-delta column
# column hinges at bottom of Story 5
node 157 $Pier1 $Floor5;
node 257 $Pier2 $Floor5;
node 357 $Pier3 $Floor5;	# zero-stiffness spring will be used on p-delta column
# column hinges at top of Story 5
node 166 $Pier1 $Floor6;
node 266 $Pier2 $Floor6;
node 366 $Pier3 $Floor6;	# zero-stiffness spring will be used on p-delta column
# column hinges at bottom of Story 6
node 167 $Pier1 $Floor6;
node 267 $Pier2 $Floor6;
node 367 $Pier3 $Floor6;	# zero-stiffness spring will be used on p-delta column
# column hinges at top of Story 6
node 176 $Pier1 $Floor7;
node 276 $Pier2 $Floor7;
node 376 $Pier3 $Floor7;	# zero-stiffness spring will be used on p-delta column


# beam hinges at Floor 2
node 122 [expr $Pier1 + $phlat23] $Floor2;
node 223 [expr $Pier2 - $phlat23] $Floor2;
# beam hinges at Floor 3
node 132 [expr $Pier1 + $phlat23] $Floor3;
node 233 [expr $Pier2 - $phlat23] $Floor3;

# beam hinges at Floor 4
node 142 [expr $Pier1 + $phlat23] $Floor4;
node 243 [expr $Pier2 - $phlat23] $Floor4;
# beam hinges at Floor 5
node 152 [expr $Pier1 + $phlat23] $Floor5;
node 253 [expr $Pier2 - $phlat23] $Floor5;
# beam hinges at Floor 6
node 162 [expr $Pier1 + $phlat23] $Floor6;
node 263 [expr $Pier2 - $phlat23] $Floor6;
# beam hinges at Floor 7
node 172 [expr $Pier1 + $phlat23] $Floor7;
node 273 [expr $Pier2 - $phlat23] $Floor7;


# constrain beam-column joints in a floor to have the same lateral displacement using the "equalDOF" command
# command: equalDOF $MasterNodeID $SlaveNodeID $dof1 $dof2...
set dof1 1;	# constrain movement in dof 1 (x-direction)
#equalDOF 12 22 $dof1;	# Floor 2:  Pier 1 to Pier 2
#equalDOF 12 32 $dof1;	# Floor 2:  Pier 1 to Pier 3
#equalDOF 13 23 $dof1;	# Floor 3:  Pier 1 to Pier 2
#equalDOF 13 33 $dof1;	# Floor 3:  Pier 1 to Pier 3
#equalDOF 14 24 $dof1;	# Floor 4:  Pier 1 to Pier 2
#equalDOF 14 34 $dof1;	# Floor 4:  Pier 1 to Pier 3
#equalDOF 15 25 $dof1;	# Floor 5:  Pier 1 to Pier 2
#equalDOF 15 35 $dof1;	# Floor 5:  Pier 1 to Pier 3
#equalDOF 16 26 $dof1;	# Floor 6:  Pier 1 to Pier 2
#equalDOF 16 36 $dof1;	# Floor 6:  Pier 1 to Pier 3
#equalDOF 17 27 $dof1;	# Floor 7:  Pier 1 to Pier 2
#equalDOF 17 37 $dof1;	# Floor 7:  Pier 1 to Pier 3


# assign boundary condidtions
# command:  fix nodeID dxFixity dyFixity rzFixity
# fixity values: 1 = constrained; 0 = unconstrained
# fix the base of the building; pin P-delta column at base
fix 11 1 1 1;
fix 21 1 1 1;
fix 317 1 1 0;	# P-delta column is pinned

equalDOF 327 326 1 2
equalDOF 337 336 1 2
equalDOF 347 346 1 2
equalDOF 357 356 1 2
equalDOF 367 366 1 2

######### Assign nodal masses #########

# RotInertia must be manually tuned as follows
# Start with a large value and adjust the brace mass inflation factor until
# the required lowest model period is obtained. Now decrease RotInertia
# until an effect is observed on the lowest modal period. This prevents
# RotInertia from being the bottleneck limiting the lowest modal period,
# instead of the brace mass.
set RotInertia 3.3

# Floor 1
mass 117 0.0 0.0 [expr {$RotInertia/50.0}]
mass 217 0.0 0.0 [expr {$RotInertia/50.0}]
mass 317 0.0 0.0 [expr {$RotInertia/50.0}]

# Floor 2
mass 12 [expr {$Floor2FrameD/$g/16.0}] [expr {$Floor2FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 122 [expr {$Floor2FrameD/$g/16.0}] [expr {$Floor2FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 126 [expr {$Floor2FrameD/$g/16.0}] [expr {$Floor2FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 127 [expr {$Floor2FrameD/$g/16.0}] [expr {$Floor2FrameD/$g/16.0}] [expr {$RotInertia/4.0}]

mass 22 [expr {$Floor2FrameD/$g/16.0}] [expr {$Floor2FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 223 [expr {$Floor2FrameD/$g/16.0}] [expr {$Floor2FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 226 [expr {$Floor2FrameD/$g/16.0}] [expr {$Floor2FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 227 [expr {$Floor2FrameD/$g/16.0}] [expr {$Floor2FrameD/$g/16.0}] [expr {$RotInertia/4.0}]

mass 42 [expr {$Floor2FrameD/$g/2.0}] [expr {$Floor2FrameD/$g/2.0}] $RotInertia

mass 326 [expr {$Floor2LeanD/$g/2.0}] [expr {$Floor2LeanD/$g/2.0}] [expr {$RotInertia/2.0}]
mass 327 [expr {$Floor2LeanD/$g/2.0}] [expr {$Floor2LeanD/$g/2.0}] [expr {$RotInertia/2.0}]

# Floor 3
mass 13 [expr {$Floor3FrameD/$g/8.0}] [expr {$Floor3FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 132 [expr {$Floor3FrameD/$g/8.0}] [expr {$Floor3FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 136 [expr {$Floor3FrameD/$g/8.0}] [expr {$Floor3FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 137 [expr {$Floor3FrameD/$g/8.0}] [expr {$Floor3FrameD/$g/8.0}] [expr {$RotInertia/4.0}]

mass 23 [expr {$Floor3FrameD/$g/8.0}] [expr {$Floor3FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 233 [expr {$Floor3FrameD/$g/8.0}] [expr {$Floor3FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 236 [expr {$Floor3FrameD/$g/8.0}] [expr {$Floor3FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 237 [expr {$Floor3FrameD/$g/8.0}] [expr {$Floor3FrameD/$g/8.0}] [expr {$RotInertia/4.0}]

mass 336 [expr {$Floor3LeanD/$g/2.0}] [expr {$Floor3LeanD/$g/2.0}] [expr {$RotInertia/2.0}]
mass 337 [expr {$Floor3LeanD/$g/2.0}] [expr {$Floor3LeanD/$g/2.0}] [expr {$RotInertia/2.0}]

# Floor 4
mass 14 [expr {$Floor4FrameD/$g/16.0}] [expr {$Floor4FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 142 [expr {$Floor4FrameD/$g/16.0}] [expr {$Floor4FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 146 [expr {$Floor4FrameD/$g/16.0}] [expr {$Floor4FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 147 [expr {$Floor4FrameD/$g/16.0}] [expr {$Floor4FrameD/$g/16.0}] [expr {$RotInertia/4.0}]

mass 24 [expr {$Floor4FrameD/$g/16.0}] [expr {$Floor4FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 243 [expr {$Floor4FrameD/$g/16.0}] [expr {$Floor4FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 246 [expr {$Floor4FrameD/$g/16.0}] [expr {$Floor4FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 247 [expr {$Floor4FrameD/$g/16.0}] [expr {$Floor4FrameD/$g/16.0}] [expr {$RotInertia/4.0}]

mass 44 [expr {$Floor4FrameD/$g/2.0}] [expr {$Floor4FrameD/$g/2.0}] $RotInertia

mass 346 [expr {$Floor4LeanD/$g/2.0}] [expr {$Floor4LeanD/$g/2.0}] [expr {$RotInertia/2.0}]
mass 347 [expr {$Floor4LeanD/$g/2.0}] [expr {$Floor4LeanD/$g/2.0}] [expr {$RotInertia/2.0}]

# Floor 5
mass 15 [expr {$Floor5FrameD/$g/8.0}] [expr {$Floor5FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 152 [expr {$Floor5FrameD/$g/8.0}] [expr {$Floor5FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 156 [expr {$Floor5FrameD/$g/8.0}] [expr {$Floor5FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 157 [expr {$Floor5FrameD/$g/8.0}] [expr {$Floor5FrameD/$g/8.0}] [expr {$RotInertia/4.0}]

mass 25 [expr {$Floor5FrameD/$g/8.0}] [expr {$Floor5FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 253 [expr {$Floor5FrameD/$g/8.0}] [expr {$Floor5FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 256 [expr {$Floor5FrameD/$g/8.0}] [expr {$Floor5FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 257 [expr {$Floor5FrameD/$g/8.0}] [expr {$Floor5FrameD/$g/8.0}] [expr {$RotInertia/4.0}]

mass 356 [expr {$Floor5LeanD/$g/2.0}] [expr {$Floor5LeanD/$g/2.0}] [expr {$RotInertia/2.0}]
mass 357 [expr {$Floor5LeanD/$g/2.0}] [expr {$Floor5LeanD/$g/2.0}] [expr {$RotInertia/2.0}]

# Floor 6
mass 16 [expr {$Floor6FrameD/$g/16.0}] [expr {$Floor6FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 162 [expr {$Floor6FrameD/$g/16.0}] [expr {$Floor6FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 166 [expr {$Floor6FrameD/$g/16.0}] [expr {$Floor6FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 167 [expr {$Floor6FrameD/$g/16.0}] [expr {$Floor6FrameD/$g/16.0}] [expr {$RotInertia/4.0}]

mass 26 [expr {$Floor6FrameD/$g/16.0}] [expr {$Floor6FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 263 [expr {$Floor6FrameD/$g/16.0}] [expr {$Floor6FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 266 [expr {$Floor6FrameD/$g/16.0}] [expr {$Floor6FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 267 [expr {$Floor6FrameD/$g/16.0}] [expr {$Floor6FrameD/$g/16.0}] [expr {$RotInertia/4.0}]

mass 46 [expr {$Floor6FrameD/$g/2.0}] [expr {$Floor6FrameD/$g/2.0}] $RotInertia

mass 366 [expr {$Floor6LeanD/$g/2.0}] [expr {$Floor6LeanD/$g/2.0}] [expr {$RotInertia/2.0}]
mass 367 [expr {$Floor6LeanD/$g/2.0}] [expr {$Floor6LeanD/$g/2.0}] [expr {$RotInertia/2.0}]

# Floor 7
mass 17 [expr {$Floor7FrameD/$g/6.0}] [expr {$Floor7FrameD/$g/6.0}] [expr {$RotInertia/3.0}]
mass 172 [expr {$Floor7FrameD/$g/6.0}] [expr {$Floor7FrameD/$g/6.0}] [expr {$RotInertia/3.0}]
mass 176 [expr {$Floor7FrameD/$g/6.0}] [expr {$Floor7FrameD/$g/6.0}] [expr {$RotInertia/3.0}]

mass 27 [expr {$Floor7FrameD/$g/6.0}] [expr {$Floor7FrameD/$g/6.0}] [expr {$RotInertia/3.0}]
mass 273 [expr {$Floor7FrameD/$g/6.0}] [expr {$Floor7FrameD/$g/6.0}] [expr {$RotInertia/3.0}]
mass 276 [expr {$Floor7FrameD/$g/6.0}] [expr {$Floor7FrameD/$g/6.0}] [expr {$RotInertia/3.0}]

mass 376 [expr {$Floor7LeanD/$g}] [expr {$Floor7LeanD/$g}] $RotInertia


###################################################################################################
#          Define Section Properties and Elements
###################################################################################################
# define material properties
set Es 29000.0;			# steel Young's modulus

####

# define column section W14x342 for Story 1 & 2
set Acol_12  101.0;		# cross-sectional area
set Icol_12  4900.0;		# moment of inertia
set Mycol_12 [expr 34480*1.17];	# yield moment, Z * Fy,exp

# define column section W14x176 for Story 3 & 4
set Acol_34  51.8;		# cross-sectional area
set Icol_34  2140.0;		# moment of inertia
set Mycol_34 [expr 16420*1.17];	# yield moment, Z * Fy,exp

# define column section W14x68 for Story 5 & 6
set Acol_56  20.0;		# cross-sectional area
set Icol_56  722.0;		# moment of inertia
set Mycol_56 [expr 5966*1.17];	# yield moment, Z * Fy,exp


# define beam section W21x62 for Floor 2
set Abeam_2  18.3;		# cross-sectional area (full section properties)
set Ibeam_2  1330.0;	# moment of inertia  (full section properties)
set Mybeam_2 [expr 144*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)

# define beam section W24x146 for Floor 3
set Abeam_3  43.0;		# cross-sectional area (full section properties)
set Ibeam_3  4580.0;	# moment of inertia  (full section properties)
set Mybeam_3 [expr 418*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)

# define beam section W18x76 for Floor 4
set Abeam_4  22.3;		# cross-sectional area (full section properties)
set Ibeam_4  1330.0;	# moment of inertia  (full section properties)
set Mybeam_4 [expr 163*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)

# define beam section W24x131 for Floor 5
set Abeam_5  38.5;		# cross-sectional area (full section properties)
set Ibeam_5  4020.0;		# moment of inertia  (full section properties)
set Mybeam_5 [expr 370*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)

# define beam section W24x104 for Floor 6
set Abeam_6  30.6;		# cross-sectional area (full section properties)
set Ibeam_6  3100.0;	# moment of inertia  (full section properties)
set Mybeam_6 [expr 289*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)

# define beam section W18x97 for Floor 7
set Abeam_7  28.5;		# cross-sectional area (full section properties)
set Ibeam_7  1750.0;	# moment of inertia  (full section properties)
set Mybeam_7 [expr 311*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)



# determine stiffness modifications to equate the stiffness of the spring-elastic element-spring subassembly to the stiffness of the actual frame member
# Reference:  Ibarra, L. F., and Krawinkler, H. (2005). "Global collapse of frame structures under seismic excitations," Technical Report 152,
#             The John A. Blume Earthquake Engineering Research Center, Department of Civil Engineering, Stanford University, Stanford, CA.
# calculate modified section properties to account for spring stiffness being in series with the elastic element stiffness
set n 10.0;		# stiffness multiplier for rotational spring

# calculate modified moment of inertia for elastic elements
set Icol_12mod  [expr $Icol_12*($n+1.0)/$n];	# modified moment of inertia for columns in Story 1 & 2
set Icol_34mod  [expr $Icol_34*($n+1.0)/$n];	# modified moment of inertia for columns in Story 3 & 4
set Icol_56mod  [expr $Icol_56*($n+1.0)/$n];	# modified moment of inertia for columns in Story 5 & 6

set Ibeam_2mod [expr $Ibeam_2*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 2
set Ibeam_3mod [expr $Ibeam_3*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 3
set Ibeam_4mod [expr $Ibeam_4*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 4
set Ibeam_5mod [expr $Ibeam_5*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 5
set Ibeam_6mod [expr $Ibeam_6*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 6
set Ibeam_7mod [expr $Ibeam_7*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 7

# calculate modified rotational stiffness for plastic hinge springs
set Ks_col_1   [expr $n*6.0*$Es*$Icol_12mod/$HStory1];		# rotational stiffness of Story 1 column springs
set Ks_col_2   [expr $n*6.0*$Es*$Icol_12mod/$HStoryTyp];	# rotational stiffness of Story 2 column springs
set Ks_col_3   [expr $n*6.0*$Es*$Icol_34mod/$HStoryTyp];	# rotational stiffness of Story 3 column springs
set Ks_col_4   [expr $n*6.0*$Es*$Icol_34mod/$HStoryTyp];	# rotational stiffness of Story 4 column springs
set Ks_col_5   [expr $n*6.0*$Es*$Icol_56mod/$HStoryTyp];	# rotational stiffness of Story 5 column springs
set Ks_col_6   [expr $n*6.0*$Es*$Icol_56mod/$HStoryTyp];	# rotational stiffness of Story 6 column springs

set Ks_beam_2 [expr $n*6.0*$Es*$Ibeam_2mod/$WBay];		# rotational stiffness of Floor 2 & 3 beam springs
set Ks_beam_3 [expr $n*6.0*$Es*$Ibeam_3mod/$WBay];		# rotational stiffness of Floor 2 & 3 beam springs
set Ks_beam_4 [expr $n*6.0*$Es*$Ibeam_4mod/$WBay];		# rotational stiffness of Floor 2 & 3 beam springs
set Ks_beam_5 [expr $n*6.0*$Es*$Ibeam_5mod/$WBay];		# rotational stiffness of Floor 2 & 3 beam springs
set Ks_beam_6 [expr $n*6.0*$Es*$Ibeam_6mod/$WBay];		# rotational stiffness of Floor 2 & 3 beam springs
set Ks_beam_7 [expr $n*6.0*$Es*$Ibeam_7mod/$WBay];		# rotational stiffness of Floor 2 & 3 beam springs


# set up geometric transformations of element
set PDeltaTransf 1;
geomTransf PDelta $PDeltaTransf; 	# PDelta transformation

# define elastic column elements using "element" command
# command: element elasticBeamColumn $eleID $iNode $jNode $A $E $I $transfID
# eleID convention:  "1xy" where 1 = col, x = Pier #, y = Story #
# Columns Story 1
element elasticBeamColumn  111  117 126 $Acol_12 $Es $Icol_12mod $PDeltaTransf;	# Pier 1
element elasticBeamColumn  121  217 226 $Acol_12 $Es $Icol_12mod $PDeltaTransf;	# Pier 2

##	element elasticBeamColumn  111  11  126 $Acol_12 $Es $Icol_12mod $PDeltaTransf;	# Pier 1
##	element elasticBeamColumn  121  21  226 $Acol_12 $Es $Icol_12mod $PDeltaTransf;	# Pier 2

# Columns Story 2
element elasticBeamColumn  112  127 136 $Acol_12 $Es $Icol_12mod $PDeltaTransf;	# Pier 1
element elasticBeamColumn  122  227 236 $Acol_12 $Es $Icol_12mod $PDeltaTransf;	# Pier 2
# Columns Story 3
element elasticBeamColumn  113  137 146 $Acol_34 $Es $Icol_34mod $PDeltaTransf;	# Pier 1
element elasticBeamColumn  123  237 246 $Acol_34 $Es $Icol_34mod $PDeltaTransf;	# Pier 2
# Columns Story 4
element elasticBeamColumn  114  147 156 $Acol_34 $Es $Icol_34mod $PDeltaTransf;	# Pier 1
element elasticBeamColumn  124  247 256 $Acol_34 $Es $Icol_34mod $PDeltaTransf;	# Pier 2
# Columns Story 5
element elasticBeamColumn  115  157 166 $Acol_56 $Es $Icol_56mod $PDeltaTransf;	# Pier 1
element elasticBeamColumn  125  257 266 $Acol_56 $Es $Icol_56mod $PDeltaTransf;	# Pier 2
# Columns Story 6
element elasticBeamColumn  116  167 176 $Acol_56 $Es $Icol_56mod $PDeltaTransf;	# Pier 1
element elasticBeamColumn  126  267 276 $Acol_56 $Es $Icol_56mod $PDeltaTransf;	# Pier 2


# define elastic beam elements
# eleID convention:  "2xy" where 2 = beam, x = Bay #, y = Floor #
# Beams Story 1
element elasticBeamColumn  212  122 42  $Abeam_2 $Es $Ibeam_2mod $PDeltaTransf;
element elasticBeamColumn  219  42  223 $Abeam_2 $Es $Ibeam_2mod $PDeltaTransf;
# Beams Story 2
element elasticBeamColumn  222  132 233 $Abeam_3 $Es $Ibeam_3mod $PDeltaTransf;

# Beams Story 3
element elasticBeamColumn  232  142 44  $Abeam_4 $Es $Ibeam_4mod $PDeltaTransf;
element elasticBeamColumn  239  44  243 $Abeam_4 $Es $Ibeam_4mod $PDeltaTransf;
# Beams Story 4
element elasticBeamColumn  242  152 253 $Abeam_5 $Es $Ibeam_5mod $PDeltaTransf;
# Beams Story 5
element elasticBeamColumn  252  162 46  $Abeam_6 $Es $Ibeam_6mod $PDeltaTransf;
element elasticBeamColumn  259  46  263 $Abeam_6 $Es $Ibeam_6mod $PDeltaTransf;
# Beams Story 6
element elasticBeamColumn  262  172 273 $Abeam_7 $Es $Ibeam_7mod $PDeltaTransf;

# define p-delta columns and rigid links
set TrussMatID 600;		# define a material ID
set Arigid 5000;		# define area of truss section (make much larger than A of frame elements)
set Aleancol 800
set Ileancol 10000
uniaxialMaterial Elastic $TrussMatID $Es;		# define truss material
# rigid links
# command: element truss $eleID $iNode $jNode $A $materialID
# eleID convention:  6xy, 6 = truss link, x = Bay #, y = Floor #
element truss 622 22 326 $Arigid $TrussMatID; # Floor 2
element truss 623 23 336 $Arigid $TrussMatID; # Floor 3

element truss 624 24 346 $Arigid $TrussMatID; # Floor 4
element truss 625 25 356 $Arigid $TrussMatID; # Floor 5
element truss 626 26 366 $Arigid $TrussMatID; # Floor 6
element truss 627 27 376 $Arigid $TrussMatID; # Floor 7

# p-delta columns
# eleID convention:  7xy, 7 = p-delta columns, x = Pier #, y = Story #
element elasticBeamColumn  731  317 326 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 1
element elasticBeamColumn  732  327 336 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 2

element elasticBeamColumn  733  337 346 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 1
element elasticBeamColumn  734  347 356 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 2
element elasticBeamColumn  735  357 366 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 1
element elasticBeamColumn  736  367 376 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 2


###################################################################################################
#          Define Rotational Springs for Plastic Hinges
###################################################################################################
# define rotational spring properties and create spring elements using "rotSpring2DModIKModel" procedure
# rotSpring2DModIKModel creates a uniaxial material spring with a bilinear response based on Modified Ibarra Krawinkler Deterioration Model
# references provided in rotSpring2DModIKModel.tcl
# input values for Story 1 column springs
set McMy 1.11;			# ratio of capping moment to yield moment, Mc / My
set LS 1000;			# basic strength deterioration (a very large # = no cyclic deterioration)
set LK 1000;			# unloading stiffness deterioration (a very large # = no cyclic deterioration)
set LA 1000;			# accelerated reloading stiffness deterioration (a very large # = no cyclic deterioration)
set LD 1000;			# post-capping strength deterioration (a very large # = no deterioration)
set cS 1.0;				# exponent for basic strength deterioration (c = 1.0 for no deterioration)
set cK 1.0;				# exponent for unloading stiffness deterioration (c = 1.0 for no deterioration)
set cA 1.0;				# exponent for accelerated reloading stiffness deterioration (c = 1.0 for no deterioration)
set cD 1.0;				# exponent for post-capping strength deterioration (c = 1.0 for no deterioration)
set th_pP 0.089;		# plastic rot capacity for pos loading
set th_pN 0.089;		# plastic rot capacity for neg loading
set th_pcP 0.743;			# post-capping rot capacity for pos loading
set th_pcN 0.743;			# post-capping rot capacity for neg loading
set ResP 0.4;			# residual strength ratio for pos loading
set ResN 0.4;			# residual strength ratio for neg loading
set th_uP 0.4;			# ultimate rot capacity for pos loading
set th_uN 0.4;			# ultimate rot capacity for neg loading
set DP 1.0;				# rate of cyclic deterioration for pos loading
set DN 1.0;				# rate of cyclic deterioration for neg loading
set a_mem [expr ($n+1.0)*($Mycol_12*($McMy-1.0)) / ($Ks_col_1*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: Eqn B.5 is incorrect)

###### define column springs
############################## Story 1 - Columns
# Spring ID: "3xya", where 3 = col spring, x = Pier #, y = Story #, a = location in story
# "a" convention: 1 = bottom of story, 2 = top of story
# command: rotSpring2DModIKModel	id    ndR  ndC     K   asPos  asNeg  MyPos      MyNeg      LS    LK    LA    LD   cS   cK   cA   cD  th_p+   th_p-   th_pc+   th_pc-  Res+   Res-   th_u+   th_u-    D+     D-
# col springs @ bottom of Story 1 (at base)

rotSpring2DModIKModel 3111 11 117 $Ks_col_1 $b $b $Mycol_12 [expr -$Mycol_12] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 3211 21 217 $Ks_col_1 $b $b $Mycol_12 [expr -$Mycol_12] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

#col springs @ top of Story 1 (below Floor 2)
rotSpring2DModIKModel 3112 12 126 $Ks_col_1 $b $b $Mycol_12 [expr -$Mycol_12] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 3212 22 226 $Ks_col_1 $b $b $Mycol_12 [expr -$Mycol_12] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

# recompute strain hardening for Story 2
set a_mem [expr ($n+1.0)*($Mycol_12*($McMy-1.0)) / ($Ks_col_2*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: there is mistake in Eqn B.5)
# col springs @ bottom of Story 2 (above Floor 2)
rotSpring2DModIKModel 3121 12 127 $Ks_col_2 $b $b $Mycol_12 [expr -$Mycol_12] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 3221 22 227 $Ks_col_2 $b $b $Mycol_12 [expr -$Mycol_12] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
#col springs @ top of Story 2 (below Floor 3)
rotSpring2DModIKModel 3122 13 136 $Ks_col_2 $b $b $Mycol_12 [expr -$Mycol_12] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 3222 23 236 $Ks_col_2 $b $b $Mycol_12 [expr -$Mycol_12] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

################################ Story 3 - Columns
# col springs @ bottom of Story 3 (above Floor 3) (USE properties from column below)
rotSpring2DModIKModel 3131 13 137 $Ks_col_2 $b $b $Mycol_12 [expr -$Mycol_12] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 3231 23 237 $Ks_col_2 $b $b $Mycol_12 [expr -$Mycol_12] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

set LS 1000;			# basic strength deterioration (a very large # = no cyclic deterioration)
set LK 1000;			# unloading stiffness deterioration (a very large # = no cyclic deterioration)
set LA 1000;			# accelerated reloading stiffness deterioration (a very large # = no cyclic deterioration)
set LD 1000;			# post-capping strength deterioration (a very large # = no deterioration)

set th_pP 0.076;		# plastic rot capacity for pos loading
set th_pN 0.076;		# plastic rot capacity for neg loading
set th_pcP 0.341;			# post-capping rot capacity for pos loading
set th_pcN 0.341;			# post-capping rot capacity for neg loading

# recompute strain hardening for Story 3
set a_mem [expr ($n+1.0)*($Mycol_34*($McMy-1.0)) / ($Ks_col_3*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: there is mistake in Eqn B.5)
#col springs @ top of Story 3 (below Floor 4)
rotSpring2DModIKModel 3132 14 146 $Ks_col_3 $b $b $Mycol_34 [expr -$Mycol_34] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 3232 24 246 $Ks_col_3 $b $b $Mycol_34 [expr -$Mycol_34] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

# recompute strain hardening for Story 4
set a_mem [expr ($n+1.0)*($Mycol_34*($McMy-1.0)) / ($Ks_col_4*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: there is mistake in Eqn B.5)
# col springs @ bottom of Story 4 (above Floor 4)
rotSpring2DModIKModel 3141 14 147 $Ks_col_4 $b $b $Mycol_34 [expr -$Mycol_34] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 3241 24 247 $Ks_col_4 $b $b $Mycol_34 [expr -$Mycol_34] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
#col springs @ top of Story 4 (below Floor 5)
rotSpring2DModIKModel 3142 15 156 $Ks_col_4 $b $b $Mycol_34 [expr -$Mycol_34] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 3242 25 256 $Ks_col_4 $b $b $Mycol_34 [expr -$Mycol_34] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

######################## Story 5 - Columns

# col springs @ bottom of Story 5 (above Floor 5) (USE properties from column below)
rotSpring2DModIKModel 3151 15 157 $Ks_col_4 $b $b $Mycol_34 [expr -$Mycol_34] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 3251 25 257 $Ks_col_4 $b $b $Mycol_34 [expr -$Mycol_34] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

set LS 1000;			# basic strength deterioration (a very large # = no cyclic deterioration)
set LK 1000;			# unloading stiffness deterioration (a very large # = no cyclic deterioration)
set LA 1000;			# accelerated reloading stiffness deterioration (a very large # = no cyclic deterioration)
set LD 1000;			# post-capping strength deterioration (a very large # = no deterioration)

set th_pP 0.063;		# plastic rot capacity for pos loading
set th_pN 0.063;		# plastic rot capacity for neg loading
set th_pcP 0.208;			# post-capping rot capacity for pos loading
set th_pcN 0.208;			# post-capping rot capacity for neg loading

# recompute strain hardening for Story 5
set a_mem [expr ($n+1.0)*($Mycol_56*($McMy-1.0)) / ($Ks_col_5*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: there is mistake in Eqn B.5)
#col springs @ top of Story 5 (below Floor 6)
rotSpring2DModIKModel 3152 16 166 $Ks_col_5 $b $b $Mycol_56 [expr -$Mycol_56] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 3252 26 266 $Ks_col_5 $b $b $Mycol_56 [expr -$Mycol_56] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

# recompute strain hardening for Story 6
set a_mem [expr ($n+1.0)*($Mycol_56*($McMy-1.0)) / ($Ks_col_6*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: there is mistake in Eqn B.5)
# col springs @ bottom of Story 6 (above Floor 6)
rotSpring2DModIKModel 3161 16 167 $Ks_col_6 $b $b $Mycol_56 [expr -$Mycol_56] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 3261 26 267 $Ks_col_6 $b $b $Mycol_56 [expr -$Mycol_56] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
#col springs @ top of Story 6 (below Floor 7)
rotSpring2DModIKModel 3162 17 176 $Ks_col_6 $b $b $Mycol_56 [expr -$Mycol_56] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 3262 27 276 $Ks_col_6 $b $b $Mycol_56 [expr -$Mycol_56] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;


# create region for frame column springs
# command: region $regionID -ele $ele_1_ID $ele_2_ID...
##	region 1 -ele 3111 3211 3112 3212 3121 3221 3122 3222 3131 3231 3132 3232 3141 3241 3142 3242 3151 3251 3152 3252 3161 3261 3162 3262 3171 3271 3172 3272;
region 1 -ele 3111 3211 3112 3212 3121 3221 3122 3222 3131 3231 3132 3232 3141 3241 3142 3242 3151 3251 3152 3252 3161 3261 3162 3262;

#############################################
# define beam springs
# Spring ID: "4xya", where 4 = beam spring, x = Bay #, y = Floor #, a = location in bay
# "a" convention: 1 = left end, 2 = right end
# redefine the rotations since they are not the same

set th_pP 0.034;
set th_pN 0.034;
set th_pcP 0.142;
set th_pcN 0.142;

# Story 2
set a_mem [expr ($n+1.0)*($Mybeam_2*($McMy-1.0)) / ($Ks_beam_2*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];								# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: there is mistake in Eqn B.5)
#beam springs at Floor 2
#	rotSpring2DModIKModel 4121 12 122 $Ks_beam_2 $b $b $Mybeam_2 [expr -$Mybeam_2] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
#	rotSpring2DModIKModel 4122 22 223 $Ks_beam_2 $b $b $Mybeam_2 [expr -$Mybeam_2] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

####### Switch to beam hinges on the 2/4/6 floors where no braces are connected
equalDOF 12 122 1 2
equalDOF 22 223 1 2

set th_pP 0.033;
set th_pN 0.033;
set th_pcP 0.181;
set th_pcN 0.181;

# Story 3
set a_mem [expr ($n+1.0)*($Mybeam_3*($McMy-1.0)) / ($Ks_beam_3*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];
#beam springs at Floor 3
rotSpring2DModIKModel 4131 13 132 $Ks_beam_3 $b $b $Mybeam_3 [expr -$Mybeam_3] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 4132 23 233 $Ks_beam_3 $b $b $Mybeam_3 [expr -$Mybeam_3] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

set th_pP 0.042;
set th_pN 0.042;
set th_pcP 0.143;
set th_pcN 0.143;

# Story 4
set a_mem [expr ($n+1.0)*($Mybeam_4*($McMy-1.0)) / ($Ks_beam_4*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];
#beam springs at Floor 4
#	rotSpring2DModIKModel 4141 14 142 $Ks_beam_4 $b $b $Mybeam_4 [expr -$Mybeam_4] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
#	rotSpring2DModIKModel 4142 24 243 $Ks_beam_4 $b $b $Mybeam_4 [expr -$Mybeam_4] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

####### Switch to beam hinges on the 2/4/6 floor where no braces are connected
equalDOF 14 142 1 2
equalDOF 24 243 1 2

set th_pP 0.032;
set th_pN 0.032;
set th_pcP 0.158;
set th_pcN 0.158;

# Story 5
set a_mem [expr ($n+1.0)*($Mybeam_5*($McMy-1.0)) / ($Ks_beam_5*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];
#beam springs at Floor 5
rotSpring2DModIKModel 4151 15 152 $Ks_beam_5 $b $b $Mybeam_5 [expr -$Mybeam_5] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 4152 25 253 $Ks_beam_5 $b $b $Mybeam_5 [expr -$Mybeam_5] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

set th_pP 0.029;
set th_pN 0.029;
set th_pcP 0.118;
set th_pcN 0.118;

# Story 6
set a_mem [expr ($n+1.0)*($Mybeam_6*($McMy-1.0)) / ($Ks_beam_6*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];
#beam springs at Floor 6
#	rotSpring2DModIKModel 4161 16 162 $Ks_beam_6 $b $b $Mybeam_6 [expr -$Mybeam_6] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
#	rotSpring2DModIKModel 4162 26 263 $Ks_beam_6 $b $b $Mybeam_6 [expr -$Mybeam_6] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

####### Switch to beam hinges on the 2/4/6 floor where no braces are connected
equalDOF 16 162 1 2
equalDOF 26 263 1 2

set th_pP 0.046;
set th_pN 0.046;
set th_pcP 0.195;
set th_pcN 0.195;

# Story 7
set a_mem [expr ($n+1.0)*($Mybeam_7*($McMy-1.0)) / ($Ks_beam_7*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];
#beam springs at Floor 7
rotSpring2DModIKModel 4171 17 172 $Ks_beam_7 $b $b $Mybeam_7 [expr -$Mybeam_7] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 4172 27 273 $Ks_beam_7 $b $b $Mybeam_7 [expr -$Mybeam_7] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

# create region for beam springs
region 2 -ele 4152 4151 4171 4172 4131 4132

#########################################################################
#########################################################################
#######Victors Addition - Braces

uniaxialMaterial Steel02 2 58.8 29800.0 0.003 20 0.925 0.15 0.0005 0.01 0.0005 0.01

set mb -0.458
set E0b 0.171
uniaxialMaterial Fatigue 200 2  -E0 $E0b -m $mb

####### Create Fatigue Material #######

##### Connections #####
## uniaxialMaterial Fatigue $matTag $tag <-E0 $E0> <-m $m> <-min $min> <-max $max>
uniaxialMaterial Steel02 3 5880.00 298000.0 0.003 20 0.925 0.15 0.0005 0.01 0.0005 0.01

#uniaxialMaterial Steel02 31 5.500 1200.0 0.00003 20 0.925 0.15 0.0005 0.01 0.0005 0.01 -0.3 1.0 -0.3 1.0 0.0
#uniaxialMaterial Steel02 32 5.500 1200.0 0.00003 20 0.925 0.15 0.0005 0.01 0.0005 0.01 -0.3 1.0 -0.3 1.0 0.0
#uniaxialMaterial Steel02 33 5.000 1200.0 0.00003 20 0.925 0.15 0.0005 0.01 0.0005 0.01 -0.3 1.0 -0.3 1.0 0.0
#uniaxialMaterial Steel02 34 4.500 1200.0 0.00003 20 0.925 0.15 0.0005 0.01 0.0005 0.01 -0.3 1.0 -0.3 1.0 0.0
#uniaxialMaterial Steel02 35 4.000 1200.0 0.00003 20 0.925 0.15 0.0005 0.01 0.0005 0.01 -0.3 1.0 -0.3 1.0 0.0
#uniaxialMaterial Steel02 36 3.000 1200.0 0.00003 20 0.925 0.15 0.0005 0.01 0.0005 0.01 -0.3 1.0 -0.3 1.0 0.0


set m -0.458
set min -2.0
set E0 0.991
### Case 1
set SH 2.95
set FailureCrit 1000.975
set FailureCrit2 1000.975
set max1 [expr (1035.0*$FailureCrit)/298000.0]
set max2 [expr (1035.0*$FailureCrit)/298000.0]
set max3 [expr (929.0*$FailureCrit)/298000.0]
set max4 [expr (788.0*$FailureCrit)/298000.0]
set max5 [expr (600.0*$FailureCrit)/298000.0]
set max6 [expr (385.0*$FailureCrit2)/298000.0]



uniaxialMaterial Fatigue 4 3 -E0 $E0 -m $m -min $min -max $max1
uniaxialMaterial Fatigue 5 3 -E0 $E0 -m $m -min $min -max $max1
uniaxialMaterial Fatigue 6 3 -E0 $E0 -m $m -min $min -max $max2
uniaxialMaterial Fatigue 7 3 -E0 $E0 -m $m -min $min -max $max2
uniaxialMaterial Fatigue 8 3 -E0 $E0 -m $m -min $min -max $max3
uniaxialMaterial Fatigue 9 3 -E0 $E0 -m $m -min $min -max $max3
uniaxialMaterial Fatigue 10 3 -E0 $E0 -m $m -min $min -max $max4
uniaxialMaterial Fatigue 11 3 -E0 $E0 -m $m -min $min -max $max4
uniaxialMaterial Fatigue 12 3 -E0 $E0 -m $m -min $min -max $max5
uniaxialMaterial Fatigue 13 3 -E0 $E0 -m $m -min $min -max $max5
uniaxialMaterial Fatigue 14 3 -E0 $E0 -m $m -min $min -max $max6
uniaxialMaterial Fatigue 15 3 -E0 $E0 -m $m -min $min -max $max6

#uniaxialMaterial Fatigue 17 32 -E0 $E0 -m $m -min $min -max $max7


set max 0.02
uniaxialMaterial Fatigue 16 3 -E0 $E0 -m $m -min $min -max $max



##### Braces #####


###### End Create Fatigue Material ######

#source HSSProperties.tcl
#source DefineGussetPlate.tcl
source [file join [file dirname [info script]] RoundHSS.tcl]

source [file join [file dirname [info script]] AlternativeBraceNodes.tcl]
source [file join [file dirname [info script]] AlternativeBraceElements.tcl]

#source DefineTrussBraces.tcl

########## Define Additional Information ##########

# Define the number of stories in the frame
set num_stories 6

# Define the control nodes used to compute story drifts
set ctrl_nodes {
    317
    326
    336
    346
    356
    366
    376
}
############################################################################
#              Gravity Loads & Gravity Analysis
############################################################################

# Recorders for gravity
 file mkdir $Output_path/gravity_drift
 file mkdir $Output_path/gravity_disp
 
# --Define the story drift recorders--
for {set story 1} {$story <= $num_stories} {incr story} {
   recorder Drift -file $Output_path/gravity_drift/gr_drift_story${story}.out -time -iNode [lindex $ctrl_nodes \
            [expr {$story - 1}]] -jNode [lindex $ctrl_nodes $story] -dof 1 -perpDirn 2
}

# Displacement
for {set story 1} {$story <= $num_stories} {incr story} {
    recorder Node -file $Output_path/gravity_disp/gr_disp_story${story}.out -time -node [lindex $ctrl_nodes $story] -dof 1 disp
}

set gravity_load_ts 1
timeSeries Linear $gravity_load_ts

set gravity_load_pattern 1
pattern Plain $gravity_load_pattern $gravity_load_ts {

load 12 0.0 [expr {-0.5*(1.05*$Floor2FrameD + 0.25*$Floor2FrameL)}] 0.0
load 22 0.0 [expr {-0.5*(1.05*$Floor2FrameD + 0.25*$Floor2FrameL)}] 0.0
load 326 0.0 [expr {-(1.05*$Floor2LeanD + 0.25*$Floor2LeanL)}] 0.0

load 13 0.0 [expr {-0.5*(1.05*$Floor3FrameD + 0.25*$Floor3FrameL)}] 0.0
load 23 0.0 [expr {-0.5*(1.05*$Floor3FrameD + 0.25*$Floor3FrameL)}] 0.0
load 336 0.0 [expr {-(1.05*$Floor3LeanD + 0.25*$Floor3LeanL)}] 0.0

load 14 0.0 [expr {-0.5*(1.05*$Floor4FrameD + 0.25*$Floor4FrameL)}] 0.0
load 24 0.0 [expr {-0.5*(1.05*$Floor4FrameD + 0.25*$Floor4FrameL)}] 0.0
load 346 0.0 [expr {-(1.05*$Floor4LeanD + 0.25*$Floor4LeanL)}] 0.0

load 15 0.0 [expr {-0.5*(1.05*$Floor5FrameD + 0.25*$Floor5FrameL)}] 0.0
load 25 0.0 [expr {-0.5*(1.05*$Floor5FrameD + 0.25*$Floor5FrameL)}] 0.0
load 356 0.0 [expr {-(1.05*$Floor5LeanD + 0.25*$Floor5LeanL)}] 0.0

load 16 0.0 [expr {-0.5*(1.05*$Floor6FrameD + 0.25*$Floor6FrameL)}] 0.0
load 26 0.0 [expr {-0.5*(1.05*$Floor6FrameD + 0.25*$Floor6FrameL)}] 0.0
load 366 0.0 [expr {-(1.05*$Floor6LeanD + 0.25*$Floor6LeanL)}] 0.0

load 17 0.0 [expr {-0.5*(1.05*$Floor7FrameD + 0.25*$Floor7FrameL)}] 0.0
load 27 0.0 [expr {-0.5*(1.05*$Floor7FrameD + 0.25*$Floor7FrameL)}] 0.0
load 376 0.0 [expr {-(1.05*$Floor7LeanD + 0.25*$Floor7LeanL)}] 0.0

}

# Gravity-analysis: load-controlled static analysis
set tol 1e-12
set maxiter 20
set print_flag 0

constraints Transformation
numberer RCM;							# renumber dof's to minimize band-width (optimization)
system SparseGEN
test RelativeEnergyIncr $tol $maxiter $print_flag
algorithm Newton;						# use Newton's solution algorithm: updates tangent stiffness at every iteration
set NstepGravity 10;					# apply gravity in 10 steps
set DGravity [expr 1.0/$NstepGravity];	# load increment
integrator LoadControl $DGravity;		# determine the next time step for an analysis
analysis Static;						# define type of analysis static or transient
if {[analyze $NstepGravity]} {
    puts "Application of gravity load failed"
}

# maintain constant gravity loads and reset time to zero
loadConst -time 0.0
wipeAnalysis
remove recorders

##############################
#	Define Damping
#############################

# Rayleigh Damping
set pi [expr {2.0*asin(1.0)}]

set eigenvalues [eigen -fullGenLapack 2]
set w1 [expr {sqrt([lindex $eigenvalues 0])}]
set w2 [expr {sqrt([lindex $eigenvalues 1])}]

# calculate damping parameters
set zeta 0.02;		# percentage of critical damping
set a0 [expr $zeta*2.0*$w1*$w2/($w1 + $w2)];	# mass damping coefficient based on first and second modes
set a1 [expr $zeta*2.0/($w1 + $w2)];			# stiffness damping coefficient based on first and second modes
set a1_mod [expr $a1*(1.0+$n)/$n];				# modified stiffness damping coefficient used for n modified elements. See Zareian & Medina 2010.

# assign damping to frame beams and columns
# command: region $regionID -eleRange $elementIDfirst $elementIDlast rayleigh $alpha_mass $alpha_currentStiff $alpha_initialStiff $alpha_committedStiff
region 4 -eleRange 111 262 rayleigh 0.0 0.0 $a1_mod 0.0;	# assign stiffness proportional damping to frame beams & columns w/ n modifications
rayleigh $a0 0.0 0.0 0.0;          				# assign mass proportional damping to structure (only assigns to nodes with mass)

region 5 -eleRange 1122100 3726700 rayleigh 0.0 0.0 $a1 0.0;	# assign stiffness proportional damping to frame braces w/ n modifications
#--------------------------------------------------------------------------------------------------------------------------------------------------------

