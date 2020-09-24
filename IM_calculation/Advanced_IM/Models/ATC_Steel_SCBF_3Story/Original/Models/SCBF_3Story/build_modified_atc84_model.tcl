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
set NStories 3;						# number of stories
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

set Floor4Weight 757.0;		# weight of Floor 4 in kips
set Floor4FrameD [expr {$Floor4Weight/12.0}]
set Floor4LeanD [expr {$Floor4Weight*11.0/12.0}]

set Floor2LL 540.0;		# live load of Floor 2 in kips
set Floor2FrameL [expr {$Floor2LL/12.0}]
set Floor2LeanL [expr {$Floor2LL*11.0/12.0}]

set Floor3LL 540.0;		# live loadof Floor 3 in kips
set Floor3FrameL [expr {$Floor3LL/12.0}]
set Floor3LeanL [expr {$Floor3LL*11.0/12.0}]

set Floor4LL 216.0;		# live load of Floor 4 in kips
set Floor4FrameL [expr {$Floor4LL/12.0}]
set Floor4LeanL [expr {$Floor4LL*11.0/12.0}]







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

node 42 [expr $Pier2/2] $Floor2;    #Extra Node to connect Braces at Center
node 44 [expr $Pier2/2] $Floor4;    #Extra Node to connect Braces at Center


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


# beam hinges at Floor 2
node 122 [expr $Pier1 + $phlat23] $Floor2;
node 223 [expr $Pier2 - $phlat23] $Floor2;
# beam hinges at Floor 3
node 132 [expr $Pier1 + $phlat23] $Floor3;
node 233 [expr $Pier2 - $phlat23] $Floor3;
# beam hinges at Floor 4
node 142 [expr $Pier1 + $phlat23] $Floor4;
node 243 [expr $Pier2 - $phlat23] $Floor4;



# assign boundary conditions
# command:  fix nodeID dxFixity dyFixity rzFixity
# fixity values: 1 = constrained; 0 = unconstrained
# fix the base of the building; pin P-delta column at base
fix 11 1 1 1;
fix 21 1 1 1;
fix 317 1 1 0;	# P-delta column is pinned

equalDOF 327 326 1 2
equalDOF 337 336 1 2

######### Assign nodal masses #########

# RotInertia must be manually tuned as follows
# Start with a large value and adjust the brace mass inflation factor until
# the required lowest model period is obtained. Now decrease RotInertia
# until an effect is observed on the lowest modal period. This prevents
# RotInertia from being the bottleneck limiting the lowest modal period,
# instead of the brace mass.
set RotInertia 1.9

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
mass 14 [expr {$Floor4FrameD/$g/12.0}] [expr {$Floor4FrameD/$g/12.0}] [expr {$RotInertia/3.0}]
mass 142 [expr {$Floor4FrameD/$g/12.0}] [expr {$Floor4FrameD/$g/12.0}] [expr {$RotInertia/3.0}]
mass 146 [expr {$Floor4FrameD/$g/12.0}] [expr {$Floor4FrameD/$g/12.0}] [expr {$RotInertia/3.0}]

mass 24 [expr {$Floor4FrameD/$g/12.0}] [expr {$Floor4FrameD/$g/12.0}] [expr {$RotInertia/3.0}]
mass 243 [expr {$Floor4FrameD/$g/12.0}] [expr {$Floor4FrameD/$g/12.0}] [expr {$RotInertia/3.0}]
mass 246 [expr {$Floor4FrameD/$g/12.0}] [expr {$Floor4FrameD/$g/12.0}] [expr {$RotInertia/3.0}]

mass 44 [expr {$Floor4FrameD/$g/2.0}] [expr {$Floor4FrameD/$g/2.0}] $RotInertia

mass 346 [expr {$Floor4LeanD/$g}] [expr {$Floor4LeanD/$g}] $RotInertia


###################################################################################################
#          Define Section Properties and Elements
###################################################################################################
# define material properties
set Es 29000.0;			# steel Young's modulus

####

# define column section W12x120 for Story 1 & 2
set Acol_12  35.3;		# cross-sectional area
set Icol_12  1070;		# moment of inertia
set Mycol_12 [expr 8023*1.17];	# yield moment, Z * Fy,exp

# define column section W12x120 for Story 3
set Acol_34  35.3;		# cross-sectional area
set Icol_34  1070;		# moment of inertia
set Mycol_34 [expr 10187*1.17];	# yield moment, Z * Fy,exp


# define beam section W18x65 for Floor 2
set Abeam_2  19.1;		# cross-sectional area (full section properties)
set Ibeam_2  1070;	# moment of inertia  (full section properties)
set Mybeam_2 [expr 133*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)

# define beam section W21x111 for Floor 3
set Abeam_3  32.7;		# cross-sectional area (full section properties)
set Ibeam_3  2670;	# moment of inertia  (full section properties)
set Mybeam_3 [expr 279*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)

# define beam section W30x173 for Floor 4
set Abeam_4  51.0;		# cross-sectional area (full section properties)
set Ibeam_4  8230;	# moment of inertia  (full section properties)
set Mybeam_4 [expr 607*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)



# determine stiffness modifications to equate the stiffness of the spring-elastic element-spring subassembly to the stiffness of the actual frame member
# Reference:  Ibarra, L. F., and Krawinkler, H. (2005). "Global collapse of frame structures under seismic excitations," Technical Report 152,
#             The John A. Blume Earthquake Engineering Research Center, Department of Civil Engineering, Stanford University, Stanford, CA.
# calculate modified section properties to account for spring stiffness being in series with the elastic element stiffness
set n 10.0;		# stiffness multiplier for rotational spring

# calculate modified moment of inertia for elastic elements
set Icol_12mod  [expr $Icol_12*($n+1.0)/$n];	# modified moment of inertia for columns in Story 1 & 2
set Icol_34mod  [expr $Icol_34*($n+1.0)/$n];	# modified moment of inertia for columns in Story 3

set Ibeam_2mod [expr $Ibeam_2*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 2
set Ibeam_3mod [expr $Ibeam_3*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 3
set Ibeam_4mod [expr $Ibeam_4*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 4

# calculate modified rotational stiffness for plastic hinge springs
set Ks_col_1   [expr $n*6.0*$Es*$Icol_12mod/$HStory1];		# rotational stiffness of Story 1 column springs
set Ks_col_2   [expr $n*6.0*$Es*$Icol_12mod/$HStoryTyp];	# rotational stiffness of Story 2 column springs
set Ks_col_3   [expr $n*6.0*$Es*$Icol_34mod/$HStoryTyp];	# rotational stiffness of Story 3 column springs

set Ks_beam_2 [expr $n*6.0*$Es*$Ibeam_2mod/$WBay];		# rotational stiffness of Floor 2 beam springs
set Ks_beam_3 [expr $n*6.0*$Es*$Ibeam_3mod/$WBay];		# rotational stiffness of Floor 3 beam springs
set Ks_beam_4 [expr $n*6.0*$Es*$Ibeam_4mod/$WBay];		# rotational stiffness of Floor 4 beam springs


# set up geometric transformations of element
set PDeltaTransf 1;
geomTransf PDelta $PDeltaTransf; 	# PDelta transformation

# define elastic column elements using "element" command
# command: element elasticBeamColumn $eleID $iNode $jNode $A $E $I $transfID
# eleID convention:  "1xy" where 1 = col, x = Pier #, y = Story #
# Columns Story 1
element elasticBeamColumn  111  117 126 $Acol_12 $Es $Icol_12mod $PDeltaTransf;	# Pier 1
element elasticBeamColumn  121  217 226 $Acol_12 $Es $Icol_12mod $PDeltaTransf;	# Pier 2

# Columns Story 2
element elasticBeamColumn  112  127 136 $Acol_12 $Es $Icol_12mod $PDeltaTransf;	# Pier 1
element elasticBeamColumn  122  227 236 $Acol_12 $Es $Icol_12mod $PDeltaTransf;	# Pier 2
# Columns Story 3
element elasticBeamColumn  113  137 146 $Acol_34 $Es $Icol_34mod $PDeltaTransf;	# Pier 1
element elasticBeamColumn  123  237 246 $Acol_34 $Es $Icol_34mod $PDeltaTransf;	# Pier 2



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


# p-delta columns
# eleID convention:  7xy, 7 = p-delta columns, x = Pier #, y = Story #
element elasticBeamColumn  731  317 326 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 1
element elasticBeamColumn  732  327 336 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 2
element elasticBeamColumn  733  337 346 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 3



###################################################################################################
#          Define Rotational Springs for Plastic Hinges
###################################################################################################
# define rotational spring properties and create spring elements using "rotSpring2DModIKModel" procedure
# rotSpring2DModIKModel creates a uniaxial material spring with a bilinear response based on Modified Ibarra Krawinkler Deterioration Model
# references provided in rotSpring2DModIKModel.tcl
# input values for Story 1 column springs
set McMy 1.1;			# ratio of capping moment to yield moment, Mc / My
set LS 1000;			# basic strength deterioration (a very large # = no cyclic deterioration)
set LK 1000;			# unloading stiffness deterioration (a very large # = no cyclic deterioration)
set LA 1000;			# accelerated reloading stiffness deterioration (a very large # = no cyclic deterioration)
set LD 1000;			# post-capping strength deterioration (a very large # = no deterioration)
set cS 1.0;				# exponent for basic strength deterioration (c = 1.0 for no deterioration)
set cK 1.0;				# exponent for unloading stiffness deterioration (c = 1.0 for no deterioration)
set cA 1.0;				# exponent for accelerated reloading stiffness deterioration (c = 1.0 for no deterioration)
set cD 1.0;				# exponent for post-capping strength deterioration (c = 1.0 for no deterioration)
set th_pP 0.081;		# plastic rot capacity for pos loading
set th_pN 0.081;		# plastic rot capacity for neg loading
set th_pcP 0.34;			# post-capping rot capacity for pos loading
set th_pcN 0.34;			# post-capping rot capacity for neg loading
set ResP 0.36;			# residual strength ratio for pos loading
set ResN 0.36;			# residual strength ratio for neg loading
set th_uP 0.2;			# ultimate rot capacity for pos loading
set th_uN 0.2;			# ultimate rot capacity for neg loading
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

set th_pP 0.09;		# plastic rot capacity for pos loading
set th_pN 0.09;		# plastic rot capacity for neg loading
set th_pcP 0.38;			# post-capping rot capacity for pos loading
set th_pcN 0.38;			# post-capping rot capacity for neg loading
set ResP 0.4;			# residual strength ratio for pos loading
set ResN 0.4;			# residual strength ratio for neg loading

# recompute strain hardening for Story 3
set a_mem [expr ($n+1.0)*($Mycol_34*($McMy-1.0)) / ($Ks_col_3*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: there is mistake in Eqn B.5)
#col springs @ top of Story 3 (below Floor 4)
rotSpring2DModIKModel 3132 14 146 $Ks_col_3 $b $b $Mycol_34 [expr -$Mycol_34] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 3232 24 246 $Ks_col_3 $b $b $Mycol_34 [expr -$Mycol_34] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;


# create region for frame column springs
# command: region $regionID -ele $ele_1_ID $ele_2_ID...
region 1 -ele 3111 3211 3112 3212 3121 3221 3122 3222 3131 3231 3132 3232;

#############################################
# define beam springs
# Spring ID: "4xya", where 4 = beam spring, x = Bay #, y = Floor #, a = location in bay
# "a" convention: 1 = left end, 2 = right end
# redefine the rotations since they are not the same

set th_pP 0.045;
set th_pN 0.045;
set th_pcP 0.21;
set th_pcN 0.21;

# Floor 2
set a_mem [expr ($n+1.0)*($Mybeam_2*($McMy-1.0)) / ($Ks_beam_2*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];								# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: there is mistake in Eqn B.5)
#beam springs at Floor 2

####### Switch to beam hinges on the 2/4 floors where no braces are connected
equalDOF 12 122 1 2
equalDOF 22 223 1 2

set th_pP 0.037;
set th_pN 0.037;
set th_pcP 0.16;
set th_pcN 0.16;

# Floor 3
set a_mem [expr ($n+1.0)*($Mybeam_3*($McMy-1.0)) / ($Ks_beam_3*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];
#beam springs at Floor 3
rotSpring2DModIKModel 4131 13 132 $Ks_beam_3 $b $b $Mybeam_3 [expr -$Mybeam_3] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 4132 23 233 $Ks_beam_3 $b $b $Mybeam_3 [expr -$Mybeam_3] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

set th_pP 0.024;
set th_pN 0.024;
set th_pcP 0.13;
set th_pcN 0.13;

# Floor 4
set a_mem [expr ($n+1.0)*($Mybeam_4*($McMy-1.0)) / ($Ks_beam_4*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];
#beam springs at Floor 4

####### Switch to beam hinges on the 2/4 floor where no braces are connected
equalDOF 14 142 1 2
equalDOF 24 243 1 2


# create region for beam springs
region 2 -ele 4131 4132

#########################################################################
#########################################################################
#######Victors Addition - Braces

uniaxialMaterial Steel02 2 58.8 29000.0 0.003 20 0.925 0.15 0.0005 0.01 0.0005 0.01

set mb -0.458
set E0b 0.171
uniaxialMaterial Fatigue 200 2  -E0 $E0b -m $mb

####### Create Fatigue Material #######

##### Connections #####
## uniaxialMaterial Fatigue $matTag $tag <-E0 $E0> <-m $m> <-min $min> <-max $max>
uniaxialMaterial Steel02 3 5880.00 290000.0 0.003 20 0.925 0.15 0.0005 0.01 0.0005 0.01

#uniaxialMaterial Steel02 31 5.500 1200.0 0.00003 20 0.925 0.15 0.0005 0.01 0.0005 0.01 -0.3 1.0 -0.3 1.0 0.0
#uniaxialMaterial Steel02 32 5.500 1200.0 0.00003 20 0.925 0.15 0.0005 0.01 0.0005 0.01 -0.3 1.0 -0.3 1.0 0.0
#uniaxialMaterial Steel02 33 5.000 1200.0 0.00003 20 0.925 0.15 0.0005 0.01 0.0005 0.01 -0.3 1.0 -0.3 1.0 0.0


#set m -0.458
#set min -2.0
#set E0 0.991
### Case 1
#set SH 2.95
#set FailureCrit 1000.975
#set FailureCrit2 1000.975
#set max1 [expr (1035.0*$FailureCrit)/298000.0]
#set max2 [expr (1035.0*$FailureCrit)/298000.0]
#set max3 [expr (929.0*$FailureCrit)/298000.0]



#uniaxialMaterial Fatigue 4 3 -E0 $E0 -m $m -min $min -max $max1
#uniaxialMaterial Fatigue 5 3 -E0 $E0 -m $m -min $min -max $max1
#uniaxialMaterial Fatigue 6 3 -E0 $E0 -m $m -min $min -max $max2
#uniaxialMaterial Fatigue 7 3 -E0 $E0 -m $m -min $min -max $max2
#uniaxialMaterial Fatigue 8 3 -E0 $E0 -m $m -min $min -max $max3
#uniaxialMaterial Fatigue 9 3 -E0 $E0 -m $m -min $min -max $max3


#set max 0.02
#uniaxialMaterial Fatigue 16 3 -E0 $E0 -m $m -min $min -max $max



##### Braces #####


###### End Create Fatigue Material ######


source [file join [file dirname [info script]] RoundHSS.tcl]

source [file join [file dirname [info script]] AlternativeBraceNodes.tcl]
source [file join [file dirname [info script]] AlternativeBraceElements.tcl]

########## Define Additional Information ##########
# Define the number of stories in the frame
set num_stories 3

# Define the control nodes used to compute story drifts
set ctrl_nodes {
    317
    326
    336
    346
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
set NstepGravity 10;					         # apply gravity in 10 steps
set DGravity [expr 1.0/$NstepGravity];	# load increment
integrator LoadControl $DGravity;		         # determine the next time step for an analysis
analysis Static;						         # define type of analysis static or transient
if {[analyze $NstepGravity]} {
    puts "Application of gravity load failed"
}

# maintain constant gravity loads and reset time to zero
loadConst -time 0.0
wipeAnalysis
remove recorders

##############################
#	Define Damping
##############################

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
region 4 -eleRange 111 239 rayleigh 0.0 0.0 $a1_mod 0.0;	# assign stiffness proportional damping to frame beams & columns w/ n modifications
rayleigh $a0 0.0 0.0 0.0;          				# assign mass proportional damping to structure (only assigns to nodes with mass)

region 5 -eleRange 1122100 3324700 rayleigh 0.0 0.0 $a1 0.0;	# assign stiffness proportional damping to frame braces w/ n modifications
#--------------------------------------------------------------------------------------------------------------------------------------------------------

