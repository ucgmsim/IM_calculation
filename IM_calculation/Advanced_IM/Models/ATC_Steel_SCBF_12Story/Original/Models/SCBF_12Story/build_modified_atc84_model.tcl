# Element and Node ID conventions:
#	1xy = frame columns with springs at both ends
#	2xy = frame beams with springs at both ends
#	6xy = trusses linking frame and P-delta column
#	7xy = P-delta columns
#	3,xya = frame column rotational springs
#	4,xya = frame beam rotational springs
#	5,xya = P-delta column rotational springs
#	where:
#		x = Pier or Bay #
#		y = Floor or Story #
#		a = an integer describing the location relative to beam-column joint (see description where elements and nodes are defined)

###################################################################################################
#          Set Up & Source Definition
###################################################################################################
model BasicBuilder -ndm 2 -ndf 3;	# Define the model builder, ndm = #dimension, ndf = #dofs
source [file join [file dirname [info script]] rotSpring2DModIKModel.tcl];

###################################################################################################
#          Define Building Geometry, Nodes, and Constraints
###################################################################################################
# define structure-geometry parameters
set NStories 12;						# number of stories
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
set Floor8  [expr $Floor7 + $HStoryTyp];
set Floor9  [expr $Floor8 + $HStoryTyp];
set Floor10 [expr $Floor9 + $HStoryTyp];
set Floor11 [expr $Floor10 + $HStoryTyp];
set Floor12 [expr $Floor11 + $HStoryTyp];
set Floor13 [expr $Floor12 + $HStoryTyp];

# calculate joint offset distance for beam plastic hinges
set phlat23 [expr 0.0];		# lateral dist from beam-col joint to loc of hinge on Floor 2

# calculate nodal masses -- lump floor masses at frame nodes
set g 386.089
set WeightTyp 986.0;
set WeightRoof 757.0;

set Floor2Weight [expr $WeightTyp/2];		# weight of Floor 2 in kips
set Floor2FrameD [expr {$Floor2Weight/6.0}]
set Floor2LeanD [expr {$Floor2Weight*5.0/6.0}]

set Floor3Weight [expr $WeightTyp/2];		# weight of Floor 3 in kips
set Floor3FrameD [expr {$Floor3Weight/6.0}]
set Floor3LeanD [expr {$Floor3Weight*5.0/6.0}]

set Floor4Weight [expr $WeightTyp/2];		# weight of Floor 4 in kips
set Floor4FrameD [expr {$Floor4Weight/6.0}]
set Floor4LeanD [expr {$Floor4Weight*5.0/6.0}]

set Floor5Weight [expr $WeightTyp/2];		# weight of Floor 5 in kips
set Floor5FrameD [expr {$Floor5Weight/6.0}]
set Floor5LeanD [expr {$Floor5Weight*5.0/6.0}]

set Floor6Weight [expr $WeightTyp/2];		# weight of Floor 6 in kips
set Floor6FrameD [expr {$Floor6Weight/6.0}]
set Floor6LeanD [expr {$Floor6Weight*5.0/6.0}]

set Floor7Weight [expr $WeightTyp/2];		# weight of Floor 7 in kips
set Floor7FrameD [expr {$Floor7Weight/6.0}]
set Floor7LeanD [expr {$Floor7Weight*5.0/6.0}]

set Floor8Weight [expr $WeightTyp/2];		# weight of Floor 8 in kips
set Floor8FrameD [expr {$Floor8Weight/6.0}]
set Floor8LeanD [expr {$Floor8Weight*5.0/6.0}]

set Floor9Weight [expr $WeightTyp/2];		# weight of Floor 9 in kips
set Floor9FrameD [expr {$Floor9Weight/6.0}]
set Floor9LeanD [expr {$Floor9Weight*5.0/6.0}]

set Floor10Weight [expr $WeightTyp/2];		# weight of Floor 10 in kips
set Floor10FrameD [expr {$Floor10Weight/6.0}]
set Floor10LeanD [expr {$Floor10Weight*5.0/6.0}]

set Floor11Weight [expr $WeightTyp/2];		# weight of Floor 11 in kips
set Floor11FrameD [expr {$Floor11Weight/6.0}]
set Floor11LeanD [expr {$Floor11Weight*5.0/6.0}]

set Floor12Weight [expr $WeightTyp/2];		# weight of Floor 12 in kips
set Floor12FrameD [expr {$Floor12Weight/6.0}]
set Floor12LeanD [expr {$Floor12Weight*5.0/6.0}]

set Floor13Weight [expr $WeightRoof/2];		# weight of Floor 13 in kips
set Floor13FrameD [expr {$Floor13Weight/6.0}]
set Floor13LeanD [expr {$Floor13Weight*5.0/6.0}]

set Floor2LL 270.0
set Floor2FrameL [expr {$Floor2LL/6.0}]
set Floor2LeanL [expr {$Floor2LL*5.0/6.0}]

set Floor3LL 270.0
set Floor3FrameL [expr {$Floor3LL/6.0}]
set Floor3LeanL [expr {$Floor3LL*5.0/6.0}]

set Floor4LL 270.0
set Floor4FrameL [expr {$Floor4LL/6.0}]
set Floor4LeanL [expr {$Floor4LL*5.0/6.0}]

set Floor5LL 270.0
set Floor5FrameL [expr {$Floor5LL/6.0}]
set Floor5LeanL [expr {$Floor5LL*5.0/6.0}]

set Floor6LL 270.0
set Floor6FrameL [expr {$Floor6LL/6.0}]
set Floor6LeanL [expr {$Floor6LL*5.0/6.0}]

set Floor7LL 270.0
set Floor7FrameL [expr {$Floor7LL/6.0}]
set Floor7LeanL [expr {$Floor7LL*5.0/6.0}]

set Floor8LL 270.0
set Floor8FrameL [expr {$Floor8LL/6.0}]
set Floor8LeanL [expr {$Floor8LL*5.0/6.0}]

set Floor9LL 270.0
set Floor9FrameL [expr {$Floor9LL/6.0}]
set Floor9LeanL [expr {$Floor9LL*5.0/6.0}]

set Floor10LL 270.0
set Floor10FrameL [expr {$Floor10LL/6.0}]
set Floor10LeanL [expr {$Floor10LL*5.0/6.0}]

set Floor11LL 270.0
set Floor11FrameL [expr {$Floor11LL/6.0}]
set Floor11LeanL [expr {$Floor11LL*5.0/6.0}]

set Floor12LL 270.0
set Floor12FrameL [expr {$Floor12LL/6.0}]
set Floor12LeanL [expr {$Floor12LL*5.0/6.0}]

set Floor13LL 108.0
set Floor13FrameL [expr {$Floor13LL/6.0}]
set Floor13LeanL [expr {$Floor13LL*5.0/6.0}]





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
node 18 $Pier1 $Floor8 
node 28 $Pier2 $Floor8 
node 19 $Pier1 $Floor9 
node 29 $Pier2 $Floor9 
node 110 $Pier1 $Floor10 
node 210 $Pier2 $Floor10 
node 111 $Pier1 $Floor11 
node 211 $Pier2 $Floor11 
node 112 $Pier1 $Floor12 
node 212 $Pier2 $Floor12 
node 113 $Pier1 $Floor13 
node 213 $Pier2 $Floor13 

node 42 [expr $Pier2/2] $Floor2;    #Extra Node to connect Braces at Center
node 44 [expr $Pier2/2] $Floor4;    #Extra Node to connect Braces at Center
node 46 [expr $Pier2/2] $Floor6;    #Extra Node to connect Braces at Center
node 48 [expr $Pier2/2] $Floor8;    #Extra Node to connect Braces at Center
node 410 [expr $Pier2/2] $Floor10;    #Extra Node to connect Braces at Center
node 412 [expr $Pier2/2] $Floor12;    #Extra Node to connect Braces at Center


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
# column hinges at bottom of Story 7
node 177 $Pier1 $Floor7;
node 277 $Pier2 $Floor7;
node 377 $Pier3 $Floor7;	# zero-stiffness spring will be used on p-delta column
# column hinges at top of Story 7
node 186 $Pier1 $Floor8;
node 286 $Pier2 $Floor8;
node 386 $Pier3 $Floor8;	# zero-stiffness spring will be used on p-delta column
# column hinges at bottom of Story 8
node 187 $Pier1 $Floor8;
node 287 $Pier2 $Floor8;
node 387 $Pier3 $Floor8;	# zero-stiffness spring will be used on p-delta column
# column hinges at top of Story 8
node 196 $Pier1 $Floor9;
node 296 $Pier2 $Floor9;
node 396 $Pier3 $Floor9;	# zero-stiffness spring will be used on p-delta column
# column hinges at bottom of Story 9
node 197 $Pier1 $Floor9;
node 297 $Pier2 $Floor9;
node 397 $Pier3 $Floor9;	# zero-stiffness spring will be used on p-delta column
# column hinges at top of Story 9
node 1106 $Pier1 $Floor10;
node 2106 $Pier2 $Floor10;
node 3106 $Pier3 $Floor10;	# zero-stiffness spring will be used on p-delta column
# column hinges at bottom of Story 10
node 1107 $Pier1 $Floor10;
node 2107 $Pier2 $Floor10;
node 3107 $Pier3 $Floor10;	# zero-stiffness spring will be used on p-delta column
# column hinges at top of Story 10
node 1116 $Pier1 $Floor11;
node 2116 $Pier2 $Floor11;
node 3116 $Pier3 $Floor11;	# zero-stiffness spring will be used on p-delta column
# column hinges at bottom of Story 11
node 1117 $Pier1 $Floor11;
node 2117 $Pier2 $Floor11;
node 3117 $Pier3 $Floor11;	# zero-stiffness spring will be used on p-delta column
# column hinges at top of Story 11
node 1126 $Pier1 $Floor12;
node 2126 $Pier2 $Floor12;
node 3126 $Pier3 $Floor12;	# zero-stiffness spring will be used on p-delta column
# column hinges at bottom of Story 12
node 1127 $Pier1 $Floor12;
node 2127 $Pier2 $Floor12;
node 3127 $Pier3 $Floor12;	# zero-stiffness spring will be used on p-delta column
# column hinges at top of Story 12
node 1136 $Pier1 $Floor13;
node 2136 $Pier2 $Floor13;
node 3136 $Pier3 $Floor13;	# zero-stiffness spring will be used on p-delta column

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
# beam hinges at Floor 8
node 182 [expr $Pier1 + $phlat23] $Floor8;
node 283 [expr $Pier2 - $phlat23] $Floor8;
# beam hinges at Floor 9
node 192 [expr $Pier1 + $phlat23] $Floor9;
node 293 [expr $Pier2 - $phlat23] $Floor9;
# beam hinges at Floor 10
node 1102 [expr $Pier1 + $phlat23] $Floor10;
node 2103 [expr $Pier2 - $phlat23] $Floor10;
# beam hinges at Floor 11
node 1112 [expr $Pier1 + $phlat23] $Floor11;
node 2113 [expr $Pier2 - $phlat23] $Floor11;
# beam hinges at Floor 12
node 1122 [expr $Pier1 + $phlat23] $Floor12;
node 2123 [expr $Pier2 - $phlat23] $Floor12;
# beam hinges at Floor 13
node 1132 [expr $Pier1 + $phlat23] $Floor13;
node 2133 [expr $Pier2 - $phlat23] $Floor13;



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
equalDOF 377 376 1 2
equalDOF 387 386 1 2
equalDOF 397 396 1 2
equalDOF 3107 3106 1 2
equalDOF 3117 3116 1 2
equalDOF 3127 3126 1 2

######### Assign nodal masses #########

# RotInertia must be manually tuned as follows
# Start with a large value and adjust the brace mass inflation factor until the required lowest model period
# is obtained. Now decrease RotInertia until an effect is observed on the lowest modal period. This prevents
# RotInertia from being the bottleneck limiting the lowest modal period, instead of the brace mass. Now
# adjust the mass deflation factor until the original fundamental period is restored.
set RotInertia 22.0
set mass_deflation_factor 0.81

# Floor 1
mass 117 0.0 0.0 [expr {$RotInertia/50.0}]
mass 217 0.0 0.0 [expr {$RotInertia/50.0}]
mass 317 0.0 0.0 [expr {$RotInertia/50.0}]

# Floor 2
mass 12 [expr {$mass_deflation_factor*$Floor2FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor2FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 122 [expr {$mass_deflation_factor*$Floor2FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor2FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 126 [expr {$mass_deflation_factor*$Floor2FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor2FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 127 [expr {$mass_deflation_factor*$Floor2FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor2FrameD/$g/16.0}] [expr {$RotInertia/4.0}]

mass 22 [expr {$mass_deflation_factor*$Floor2FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor2FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 223 [expr {$mass_deflation_factor*$Floor2FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor2FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 226 [expr {$mass_deflation_factor*$Floor2FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor2FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 227 [expr {$mass_deflation_factor*$Floor2FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor2FrameD/$g/16.0}] [expr {$RotInertia/4.0}]

mass 42 [expr {$mass_deflation_factor*$Floor2FrameD/$g/2.0}] [expr {$mass_deflation_factor*$Floor2FrameD/$g/2.0}] $RotInertia

mass 326 [expr {$mass_deflation_factor*$Floor2LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor2LeanD/$g/2.0}] [expr {$RotInertia/2.0}]
mass 327 [expr {$mass_deflation_factor*$Floor2LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor2LeanD/$g/2.0}] [expr {$RotInertia/2.0}]

# Floor 3
mass 13 [expr {$mass_deflation_factor*$Floor3FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor3FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 132 [expr {$mass_deflation_factor*$Floor3FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor3FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 136 [expr {$mass_deflation_factor*$Floor3FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor3FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 137 [expr {$mass_deflation_factor*$Floor3FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor3FrameD/$g/8.0}] [expr {$RotInertia/4.0}]

mass 23 [expr {$mass_deflation_factor*$Floor3FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor3FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 233 [expr {$mass_deflation_factor*$Floor3FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor3FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 236 [expr {$mass_deflation_factor*$Floor3FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor3FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 237 [expr {$mass_deflation_factor*$Floor3FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor3FrameD/$g/8.0}] [expr {$RotInertia/4.0}]

mass 336 [expr {$mass_deflation_factor*$Floor3LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor3LeanD/$g/2.0}] [expr {$RotInertia/2.0}]
mass 337 [expr {$mass_deflation_factor*$Floor3LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor3LeanD/$g/2.0}] [expr {$RotInertia/2.0}]

# Floor 4
mass 14 [expr {$mass_deflation_factor*$Floor4FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor4FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 142 [expr {$mass_deflation_factor*$Floor4FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor4FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 146 [expr {$mass_deflation_factor*$Floor4FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor4FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 147 [expr {$mass_deflation_factor*$Floor4FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor4FrameD/$g/16.0}] [expr {$RotInertia/4.0}]

mass 24 [expr {$mass_deflation_factor*$Floor4FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor4FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 243 [expr {$mass_deflation_factor*$Floor4FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor4FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 246 [expr {$mass_deflation_factor*$Floor4FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor4FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 247 [expr {$mass_deflation_factor*$Floor4FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor4FrameD/$g/16.0}] [expr {$RotInertia/4.0}]

mass 44 [expr {$mass_deflation_factor*$Floor4FrameD/$g/2.0}] [expr {$mass_deflation_factor*$Floor4FrameD/$g/2.0}] $RotInertia

mass 346 [expr {$mass_deflation_factor*$Floor4LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor4LeanD/$g/2.0}] [expr {$RotInertia/2.0}]
mass 347 [expr {$mass_deflation_factor*$Floor4LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor4LeanD/$g/2.0}] [expr {$RotInertia/2.0}]

# Floor 5
mass 15 [expr {$mass_deflation_factor*$Floor5FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor5FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 152 [expr {$mass_deflation_factor*$Floor5FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor5FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 156 [expr {$mass_deflation_factor*$Floor5FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor5FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 157 [expr {$mass_deflation_factor*$Floor5FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor5FrameD/$g/8.0}] [expr {$RotInertia/4.0}]

mass 25 [expr {$mass_deflation_factor*$Floor5FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor5FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 253 [expr {$mass_deflation_factor*$Floor5FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor5FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 256 [expr {$mass_deflation_factor*$Floor5FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor5FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 257 [expr {$mass_deflation_factor*$Floor5FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor5FrameD/$g/8.0}] [expr {$RotInertia/4.0}]

mass 356 [expr {$mass_deflation_factor*$Floor5LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor5LeanD/$g/2.0}] [expr {$RotInertia/2.0}]
mass 357 [expr {$mass_deflation_factor*$Floor5LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor5LeanD/$g/2.0}] [expr {$RotInertia/2.0}]

# Floor 6
mass 16 [expr {$mass_deflation_factor*$Floor6FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor6FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 162 [expr {$mass_deflation_factor*$Floor6FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor6FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 166 [expr {$mass_deflation_factor*$Floor6FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor6FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 167 [expr {$mass_deflation_factor*$Floor6FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor6FrameD/$g/16.0}] [expr {$RotInertia/4.0}]

mass 26 [expr {$mass_deflation_factor*$Floor6FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor6FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 263 [expr {$mass_deflation_factor*$Floor6FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor6FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 266 [expr {$mass_deflation_factor*$Floor6FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor6FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 267 [expr {$mass_deflation_factor*$Floor6FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor6FrameD/$g/16.0}] [expr {$RotInertia/4.0}]

mass 46 [expr {$mass_deflation_factor*$Floor6FrameD/$g/2.0}] [expr {$mass_deflation_factor*$Floor6FrameD/$g/2.0}] $RotInertia

mass 366 [expr {$mass_deflation_factor*$Floor6LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor6LeanD/$g/2.0}] [expr {$RotInertia/2.0}]
mass 367 [expr {$mass_deflation_factor*$Floor6LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor6LeanD/$g/2.0}] [expr {$RotInertia/2.0}]

# Floor 7
mass 17 [expr {$mass_deflation_factor*$Floor7FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor7FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 172 [expr {$mass_deflation_factor*$Floor7FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor7FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 176 [expr {$mass_deflation_factor*$Floor7FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor7FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 177 [expr {$mass_deflation_factor*$Floor7FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor7FrameD/$g/8.0}] [expr {$RotInertia/4.0}]

mass 27 [expr {$mass_deflation_factor*$Floor7FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor7FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 273 [expr {$mass_deflation_factor*$Floor7FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor7FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 276 [expr {$mass_deflation_factor*$Floor7FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor7FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 277 [expr {$mass_deflation_factor*$Floor7FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor7FrameD/$g/8.0}] [expr {$RotInertia/4.0}]

mass 376 [expr {$mass_deflation_factor*$Floor7LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor7LeanD/$g/2.0}] [expr {$RotInertia/2.0}]
mass 377 [expr {$mass_deflation_factor*$Floor7LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor7LeanD/$g/2.0}] [expr {$RotInertia/2.0}]

# Floor 8
mass 18 [expr {$mass_deflation_factor*$Floor8FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor8FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 182 [expr {$mass_deflation_factor*$Floor8FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor8FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 186 [expr {$mass_deflation_factor*$Floor8FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor8FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 187 [expr {$mass_deflation_factor*$Floor8FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor8FrameD/$g/16.0}] [expr {$RotInertia/4.0}]

mass 28 [expr {$mass_deflation_factor*$Floor8FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor8FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 283 [expr {$mass_deflation_factor*$Floor8FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor8FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 286 [expr {$mass_deflation_factor*$Floor8FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor8FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 287 [expr {$mass_deflation_factor*$Floor8FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor8FrameD/$g/16.0}] [expr {$RotInertia/4.0}]

mass 48 [expr {$mass_deflation_factor*$Floor8FrameD/$g/2.0}] [expr {$mass_deflation_factor*$Floor8FrameD/$g/2.0}] $RotInertia

mass 386 [expr {$mass_deflation_factor*$Floor8LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor8LeanD/$g/2.0}] [expr {$RotInertia/2.0}]
mass 387 [expr {$mass_deflation_factor*$Floor8LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor8LeanD/$g/2.0}] [expr {$RotInertia/2.0}]

# Floor 9
mass 19 [expr {$mass_deflation_factor*$Floor9FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor9FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 192 [expr {$mass_deflation_factor*$Floor9FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor9FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 196 [expr {$mass_deflation_factor*$Floor9FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor9FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 197 [expr {$mass_deflation_factor*$Floor9FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor9FrameD/$g/8.0}] [expr {$RotInertia/4.0}]

mass 29 [expr {$mass_deflation_factor*$Floor9FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor9FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 293 [expr {$mass_deflation_factor*$Floor9FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor9FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 296 [expr {$mass_deflation_factor*$Floor9FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor9FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 297 [expr {$mass_deflation_factor*$Floor9FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor9FrameD/$g/8.0}] [expr {$RotInertia/4.0}]

mass 396 [expr {$mass_deflation_factor*$Floor9LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor9LeanD/$g/2.0}] [expr {$RotInertia/2.0}]
mass 397 [expr {$mass_deflation_factor*$Floor9LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor9LeanD/$g/2.0}] [expr {$RotInertia/2.0}]

# Floor 10
mass 110 [expr {$mass_deflation_factor*$Floor10FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor10FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 1102 [expr {$mass_deflation_factor*$Floor10FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor10FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 1106 [expr {$mass_deflation_factor*$Floor10FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor10FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 1107 [expr {$mass_deflation_factor*$Floor10FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor10FrameD/$g/16.0}] [expr {$RotInertia/4.0}]

mass 210 [expr {$mass_deflation_factor*$Floor10FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor10FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 2103 [expr {$mass_deflation_factor*$Floor10FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor10FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 2106 [expr {$mass_deflation_factor*$Floor10FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor10FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 2107 [expr {$mass_deflation_factor*$Floor10FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor10FrameD/$g/16.0}] [expr {$RotInertia/4.0}]

mass 410 [expr {$mass_deflation_factor*$Floor10FrameD/$g/2.0}] [expr {$mass_deflation_factor*$Floor10FrameD/$g/2.0}] $RotInertia

mass 3106 [expr {$mass_deflation_factor*$Floor10LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor10LeanD/$g/2.0}] [expr {$RotInertia/2.0}]
mass 3107 [expr {$mass_deflation_factor*$Floor10LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor10LeanD/$g/2.0}] [expr {$RotInertia/2.0}]

# Floor 11
mass 111 [expr {$mass_deflation_factor*$Floor11FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor11FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 1112 [expr {$mass_deflation_factor*$Floor11FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor11FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 1116 [expr {$mass_deflation_factor*$Floor11FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor11FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 1117 [expr {$mass_deflation_factor*$Floor11FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor11FrameD/$g/8.0}] [expr {$RotInertia/4.0}]

mass 211 [expr {$mass_deflation_factor*$Floor11FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor11FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 2113 [expr {$mass_deflation_factor*$Floor11FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor11FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 2116 [expr {$mass_deflation_factor*$Floor11FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor11FrameD/$g/8.0}] [expr {$RotInertia/4.0}]
mass 2117 [expr {$mass_deflation_factor*$Floor11FrameD/$g/8.0}] [expr {$mass_deflation_factor*$Floor11FrameD/$g/8.0}] [expr {$RotInertia/4.0}]

mass 3116 [expr {$mass_deflation_factor*$Floor11LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor11LeanD/$g/2.0}] [expr {$RotInertia/2.0}]
mass 3117 [expr {$mass_deflation_factor*$Floor11LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor11LeanD/$g/2.0}] [expr {$RotInertia/2.0}]

# Floor 12
mass 112 [expr {$mass_deflation_factor*$Floor12FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor12FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 1122 [expr {$mass_deflation_factor*$Floor12FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor12FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 1126 [expr {$mass_deflation_factor*$Floor12FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor12FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 1127 [expr {$mass_deflation_factor*$Floor12FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor12FrameD/$g/16.0}] [expr {$RotInertia/4.0}]

mass 212 [expr {$mass_deflation_factor*$Floor12FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor12FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 2123 [expr {$mass_deflation_factor*$Floor12FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor12FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 2126 [expr {$mass_deflation_factor*$Floor12FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor12FrameD/$g/16.0}] [expr {$RotInertia/4.0}]
mass 2127 [expr {$mass_deflation_factor*$Floor12FrameD/$g/16.0}] [expr {$mass_deflation_factor*$Floor12FrameD/$g/16.0}] [expr {$RotInertia/4.0}]

mass 412 [expr {$mass_deflation_factor*$Floor12FrameD/$g/2.0}] [expr {$mass_deflation_factor*$Floor12FrameD/$g/2.0}] $RotInertia

mass 3126 [expr {$mass_deflation_factor*$Floor12LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor12LeanD/$g/2.0}] [expr {$RotInertia/2.0}]
mass 3127 [expr {$mass_deflation_factor*$Floor12LeanD/$g/2.0}] [expr {$mass_deflation_factor*$Floor12LeanD/$g/2.0}] [expr {$RotInertia/2.0}]

# Floor 13
mass 113 [expr {$mass_deflation_factor*$Floor13FrameD/$g/6.0}] [expr {$mass_deflation_factor*$Floor13FrameD/$g/6.0}] [expr {$RotInertia/3.0}]
mass 1132 [expr {$mass_deflation_factor*$Floor13FrameD/$g/6.0}] [expr {$mass_deflation_factor*$Floor13FrameD/$g/6.0}] [expr {$RotInertia/3.0}]
mass 1136 [expr {$mass_deflation_factor*$Floor13FrameD/$g/6.0}] [expr {$mass_deflation_factor*$Floor13FrameD/$g/6.0}] [expr {$RotInertia/3.0}]

mass 213 [expr {$mass_deflation_factor*$Floor13FrameD/$g/6.0}] [expr {$mass_deflation_factor*$Floor13FrameD/$g/6.0}] [expr {$RotInertia/3.0}]
mass 2133 [expr {$mass_deflation_factor*$Floor13FrameD/$g/6.0}] [expr {$mass_deflation_factor*$Floor13FrameD/$g/6.0}] [expr {$RotInertia/3.0}]
mass 2136 [expr {$mass_deflation_factor*$Floor13FrameD/$g/6.0}] [expr {$mass_deflation_factor*$Floor13FrameD/$g/6.0}] [expr {$RotInertia/3.0}]

mass 3136 [expr {$mass_deflation_factor*$Floor13LeanD/$g}] [expr {$mass_deflation_factor*$Floor13LeanD/$g}] $RotInertia


###################################################################################################
#          Define Section Properties and Elements
###################################################################################################
# define material properties
set Es 29000.0;			# steel Young's modulus

# define column section W14x550 for Story 1 & 2
set Acol_12  162;		# cross-sectional area
set Icol_12  9430;		# moment of inertia
set Mycol_12 [expr 62156*1.17];	# yield moment, Z * Fy,exp

# define column section W14x398 for Story 3 & 4
set Acol_34  117;		# cross-sectional area
set Icol_34  6000;		# moment of inertia
set Mycol_34 [expr 42090*1.17];	# yield moment, Z * Fy,exp

# define column section W14x283 for Story 5 & 6
set Acol_56  83.3;		# cross-sectional area
set Icol_56  3840.0;		# moment of inertia
set Mycol_56 [expr 28373*1.17];	# yield moment, Z * Fy,exp

# define column section W14x193 for Story 7 & 8
set Acol_78  56.8;		# cross-sectional area
set Icol_78  2400;		# moment of inertia
set Mycol_78 [expr 18525*1.17];	# yield moment, Z * Fy,exp

# define column section W14x99 for Story 9 & 10
set Acol_910  29.1;		# cross-sectional area
set Icol_910  1110;		# moment of inertia
set Mycol_910 [expr 8825*1.17];	# yield moment, Z * Fy,exp

# define column section W12x45 for Story 11 & 12
set Acol_1112  13.1;		# cross-sectional area
set Icol_1112  348;		# moment of inertia
set Mycol_1112 [expr 3360*1.17];	# yield moment, Z * Fy,exp


set Amult 1; #Doesn't serve any purpose anymore.

# define beam section W18x35 for Floor 2
set Abeam_2  [expr $Amult * 10.3];		# cross-sectional area (full section properties)
set Ibeam_2  510.0;	# moment of inertia  (full section properties)
set Mybeam_2 [expr 66.5*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)

# define beam section W18x71 for Floor 3
set Abeam_3  [expr $Amult * 20.8];		# cross-sectional area (full section properties)
set Ibeam_3  1170;	# moment of inertia  (full section properties)
set Mybeam_3 [expr 146*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)

# define beam section W18x35 for Floor 4
set Abeam_4  [expr $Amult * 10.3];		# cross-sectional area (full section properties)
set Ibeam_4  510.0;	# moment of inertia  (full section properties)
set Mybeam_4 [expr 66.5*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)

# define beam section W18x71 for Floor 5
set Abeam_5  [expr $Amult * 20.8];		# cross-sectional area (full section properties)
set Ibeam_5  1170;		# moment of inertia  (full section properties)
set Mybeam_5 [expr 146*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)

# define beam section W18x35 for Floor 6
set Abeam_6  [expr $Amult * 10.3];		# cross-sectional area (full section properties)
set Ibeam_6  510.0;	# moment of inertia  (full section properties)
set Mybeam_6 [expr 66.5*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)

# define beam section W18x65 for Floor 7
set Abeam_7  [expr $Amult * 19.1];		# cross-sectional area (full section properties)
set Ibeam_7  1070;	# moment of inertia  (full section properties)
set Mybeam_7 [expr 133*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)

# define beam section W18x35 for Floor 8
set Abeam_8  [expr $Amult * 10.3];		# cross-sectional area (full section properties)
set Ibeam_8  510.0;	# moment of inertia  (full section properties)
set Mybeam_8 [expr 66.5*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)

# define beam section W18x65 for Floor 9
set Abeam_9  [expr $Amult * 19.1];		# cross-sectional area (full section properties)
set Ibeam_9  1070;	# moment of inertia  (full section properties)
set Mybeam_9 [expr 133*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)

# define beam section W18x35 for Floor 10
set Abeam_10  [expr $Amult * 10.3];		# cross-sectional area (full section properties)
set Ibeam_10  510.0;	# moment of inertia  (full section properties)
set Mybeam_10 [expr 66.5*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)

# define beam section W18x60 for Floor 11
set Abeam_11  [expr $Amult * 17.6];		# cross-sectional area (full section properties)
set Ibeam_11  984;	# moment of inertia  (full section properties)
set Mybeam_11 [expr 123*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)

# define beam section W18x35 for Floor 12
set Abeam_12  [expr $Amult * 10.3];		# cross-sectional area (full section properties)
set Ibeam_12  510.0;	# moment of inertia  (full section properties)
set Mybeam_12 [expr 66.5*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)

# define beam section W18x55 for Floor 13
set Abeam_13  [expr $Amult * 16.2];		# cross-sectional area (full section properties)
set Ibeam_13  890;	# moment of inertia  (full section properties)
set Mybeam_13 [expr 112*55*1.17];	# yield moment at plastic hinge location (i.e., My of RBS section, if used)





# determine stiffness modifications to equate the stiffness of the spring-elastic element-spring subassembly to the stiffness of the actual frame member
# Reference:  Ibarra, L. F., and Krawinkler, H. (2005). "Global collapse of frame structures under seismic excitations," Technical Report 152,
#             The John A. Blume Earthquake Engineering Research Center, Department of Civil Engineering, Stanford University, Stanford, CA.
# calculate modified section properties to account for spring stiffness being in series with the elastic element stiffness
set n 10.0;		# stiffness multiplier for rotational spring

# calculate modified moment of inertia for elastic elements
set Icol_12mod  [expr $Icol_12*($n+1.0)/$n];	# modified moment of inertia for columns in Story 1 & 2
set Icol_34mod  [expr $Icol_34*($n+1.0)/$n];	# modified moment of inertia for columns in Story 3 & 4
set Icol_56mod  [expr $Icol_56*($n+1.0)/$n];	# modified moment of inertia for columns in Story 5 & 6
set Icol_78mod  [expr $Icol_78*($n+1.0)/$n];	# modified moment of inertia for columns in Story 7 & 8
set Icol_910mod  [expr $Icol_910*($n+1.0)/$n];	# modified moment of inertia for columns in Story 9 & 10
set Icol_1112mod  [expr $Icol_1112*($n+1.0)/$n];	# modified moment of inertia for columns in Story 11 & 12

set Ibeam_2mod [expr $Ibeam_2*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 2
set Ibeam_3mod [expr $Ibeam_3*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 3
set Ibeam_4mod [expr $Ibeam_4*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 4
set Ibeam_5mod [expr $Ibeam_5*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 5
set Ibeam_6mod [expr $Ibeam_6*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 6
set Ibeam_7mod [expr $Ibeam_7*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 7
set Ibeam_8mod [expr $Ibeam_8*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 8
set Ibeam_9mod [expr $Ibeam_9*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 9
set Ibeam_10mod [expr $Ibeam_10*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 10
set Ibeam_11mod [expr $Ibeam_11*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 11
set Ibeam_12mod [expr $Ibeam_12*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 12
set Ibeam_13mod [expr $Ibeam_13*($n+1.0)/$n];	# modified moment of inertia for beams in Floor 13


# calculate modified rotational stiffness for plastic hinge springs
set Ks_col_1   [expr $n*6.0*$Es*$Icol_12mod/$HStory1];		# rotational stiffness of Story 1 column springs
set Ks_col_2   [expr $n*6.0*$Es*$Icol_12mod/$HStoryTyp];	# rotational stiffness of Story 2 column springs
set Ks_col_3   [expr $n*6.0*$Es*$Icol_34mod/$HStoryTyp];	# rotational stiffness of Story 3 column springs
set Ks_col_4   [expr $n*6.0*$Es*$Icol_34mod/$HStoryTyp];	# rotational stiffness of Story 4 column springs
set Ks_col_5   [expr $n*6.0*$Es*$Icol_56mod/$HStoryTyp];	# rotational stiffness of Story 5 column springs
set Ks_col_6   [expr $n*6.0*$Es*$Icol_56mod/$HStoryTyp];	# rotational stiffness of Story 6 column springs
set Ks_col_7   [expr $n*6.0*$Es*$Icol_78mod/$HStoryTyp];	# rotational stiffness of Story 7 column springs
set Ks_col_8   [expr $n*6.0*$Es*$Icol_78mod/$HStoryTyp];	# rotational stiffness of Story 8 column springs
set Ks_col_9   [expr $n*6.0*$Es*$Icol_910mod/$HStoryTyp];	# rotational stiffness of Story 9 column springs
set Ks_col_10   [expr $n*6.0*$Es*$Icol_910mod/$HStoryTyp];	# rotational stiffness of Story 10 column springs
set Ks_col_11   [expr $n*6.0*$Es*$Icol_1112mod/$HStoryTyp];	# rotational stiffness of Story 11 column springs
set Ks_col_12   [expr $n*6.0*$Es*$Icol_1112mod/$HStoryTyp];	# rotational stiffness of Story 12 column springs


set Ks_beam_2 [expr $n*6.0*$Es*$Ibeam_2mod/$WBay];		# rotational stiffness of Floor 2 beam springs
set Ks_beam_3 [expr $n*6.0*$Es*$Ibeam_3mod/$WBay];		# rotational stiffness of Floor 3 beam springs
set Ks_beam_4 [expr $n*6.0*$Es*$Ibeam_4mod/$WBay];		# rotational stiffness of Floor 4 beam springs
set Ks_beam_5 [expr $n*6.0*$Es*$Ibeam_5mod/$WBay];		# rotational stiffness of Floor 5 beam springs
set Ks_beam_6 [expr $n*6.0*$Es*$Ibeam_6mod/$WBay];		# rotational stiffness of Floor 6 beam springs
set Ks_beam_7 [expr $n*6.0*$Es*$Ibeam_7mod/$WBay];		# rotational stiffness of Floor 7 beam springs
set Ks_beam_8 [expr $n*6.0*$Es*$Ibeam_8mod/$WBay];		# rotational stiffness of Floor 8 beam springs
set Ks_beam_9 [expr $n*6.0*$Es*$Ibeam_9mod/$WBay];		# rotational stiffness of Floor 9 beam springs
set Ks_beam_10 [expr $n*6.0*$Es*$Ibeam_10mod/$WBay];		# rotational stiffness of Floor 10 beam springs
set Ks_beam_11 [expr $n*6.0*$Es*$Ibeam_11mod/$WBay];		# rotational stiffness of Floor 11 beam springs
set Ks_beam_12 [expr $n*6.0*$Es*$Ibeam_12mod/$WBay];		# rotational stiffness of Floor 12 beam springs
set Ks_beam_13 [expr $n*6.0*$Es*$Ibeam_13mod/$WBay];		# rotational stiffness of Floor 13 beam springs


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
# Columns Story 4
element elasticBeamColumn  114  147 156 $Acol_34 $Es $Icol_34mod $PDeltaTransf;	# Pier 1
element elasticBeamColumn  124  247 256 $Acol_34 $Es $Icol_34mod $PDeltaTransf;	# Pier 2
# Columns Story 5
element elasticBeamColumn  115  157 166 $Acol_56 $Es $Icol_56mod $PDeltaTransf;	# Pier 1
element elasticBeamColumn  125  257 266 $Acol_56 $Es $Icol_56mod $PDeltaTransf;	# Pier 2
# Columns Story 6
element elasticBeamColumn  116  167 176 $Acol_56 $Es $Icol_56mod $PDeltaTransf;	# Pier 1
element elasticBeamColumn  126  267 276 $Acol_56 $Es $Icol_56mod $PDeltaTransf;	# Pier 2
# Columns Story 7
element elasticBeamColumn  117  177 186 $Acol_78 $Es $Icol_78mod $PDeltaTransf;	# Pier 1
element elasticBeamColumn  127  277 286 $Acol_78 $Es $Icol_78mod $PDeltaTransf;	# Pier 2
# Columns Story 8
element elasticBeamColumn  118  187 196 $Acol_78 $Es $Icol_78mod $PDeltaTransf;	# Pier 1
element elasticBeamColumn  128  287 296 $Acol_78 $Es $Icol_78mod $PDeltaTransf;	# Pier 2
# Columns Story 9
element elasticBeamColumn  119  197 1106 $Acol_910 $Es $Icol_910mod $PDeltaTransf;	# Pier 1
element elasticBeamColumn  129  297 2106 $Acol_910 $Es $Icol_910mod $PDeltaTransf;	# Pier 2
# Columns Story 10
element elasticBeamColumn  1110  1107 1116 $Acol_910 $Es $Icol_910mod $PDeltaTransf;	# Pier 1
element elasticBeamColumn  1210  2107 2116 $Acol_910 $Es $Icol_910mod $PDeltaTransf;	# Pier 2
# Columns Story 11
element elasticBeamColumn  1111  1117 1126 $Acol_1112 $Es $Icol_1112mod $PDeltaTransf;	# Pier 1
element elasticBeamColumn  1211  2117 2126 $Acol_1112 $Es $Icol_1112mod $PDeltaTransf;	# Pier 2
# Columns Story 12
element elasticBeamColumn  1112  1127 1136 $Acol_1112 $Es $Icol_1112mod $PDeltaTransf;	# Pier 1
element elasticBeamColumn  1212  2127 2136 $Acol_1112 $Es $Icol_1112mod $PDeltaTransf;	# Pier 2

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
# Beams Story 7
element elasticBeamColumn  272  182 48  $Abeam_8 $Es $Ibeam_8mod $PDeltaTransf;
element elasticBeamColumn  279  48  283 $Abeam_8 $Es $Ibeam_8mod $PDeltaTransf;
# Beams Story 8
element elasticBeamColumn  282  192 293 $Abeam_9 $Es $Ibeam_9mod $PDeltaTransf;
# Beams Story 9
element elasticBeamColumn  292  1102 410  $Abeam_10 $Es $Ibeam_10mod $PDeltaTransf;
element elasticBeamColumn  299  410  2103 $Abeam_10 $Es $Ibeam_10mod $PDeltaTransf;
# Beams Story 10
element elasticBeamColumn  2102  1112 2113 $Abeam_11 $Es $Ibeam_11mod $PDeltaTransf;
# Beams Story 11
element elasticBeamColumn  2112  1122 412  $Abeam_12 $Es $Ibeam_12mod $PDeltaTransf;
element elasticBeamColumn  2119  412  2123 $Abeam_12 $Es $Ibeam_12mod $PDeltaTransf;
# Beams Story 12
element elasticBeamColumn  2122  1132 2133 $Abeam_13 $Es $Ibeam_13mod $PDeltaTransf;


# define p-delta columns and rigid links
set TrussMatID 600;		# define a material ID
set Arigid 5000;		# define area of truss section (make much larger than A of frame elements)
set Aleancol 800
set Ileancol 10000
uniaxialMaterial Elastic $TrussMatID $Es;		# define truss material

# rigid links
# command: element truss $eleID $iNode $jNode $A $materialID
# eleID convention:  6xy, 6 = truss link, x = Bay #, y = Floor #
element truss 6202 22 326 $Arigid $TrussMatID; # Floor 2
element truss 6203 23 336 $Arigid $TrussMatID; # Floor 3
element truss 6204 24 346 $Arigid $TrussMatID; # Floor 4
element truss 6205 25 356 $Arigid $TrussMatID; # Floor 5
element truss 6206 26 366 $Arigid $TrussMatID; # Floor 6
element truss 6207 27 376 $Arigid $TrussMatID; # Floor 7
element truss 6208 28 386 $Arigid $TrussMatID; # Floor 8
element truss 6209 29 396 $Arigid $TrussMatID; # Floor 9
element truss 6210 210 3106 $Arigid $TrussMatID; # Floor 10
element truss 6211 211 3116 $Arigid $TrussMatID; # Floor 11
element truss 6212 212 3126 $Arigid $TrussMatID; # Floor 12
element truss 6213 213 3136 $Arigid $TrussMatID; # Floor 13


# p-delta columns
# eleID convention:  7xy, 7 = p-delta columns, x = Pier #, y = Story #
element elasticBeamColumn  7301  317 326 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 1
element elasticBeamColumn  7302  327 336 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 2
element elasticBeamColumn  7303  337 346 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 3
element elasticBeamColumn  7304  347 356 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 4
element elasticBeamColumn  7305  357 366 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 5
element elasticBeamColumn  7306  367 376 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 6
element elasticBeamColumn  7307  377 386 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 7
element elasticBeamColumn  7308  387 396 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 8
element elasticBeamColumn  7309  397 3106 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 9
element elasticBeamColumn  7310  3107 3116 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 10
element elasticBeamColumn  7311  3117 3126 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 11
element elasticBeamColumn  7312  3127 3136 $Aleancol $Es $Ileancol $PDeltaTransf;	# Story 12


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
set th_pP 0.095;		# plastic rot capacity for pos loading
set th_pN 0.095;		# plastic rot capacity for neg loading
set th_pcP 1.243;			# post-capping rot capacity for pos loading
set th_pcN 1.243;			# post-capping rot capacity for neg loading
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

set th_pP 0.091;		# plastic rot capacity for pos loading
set th_pN 0.091;		# plastic rot capacity for neg loading
set th_pcP 0.878;			# post-capping rot capacity for pos loading
set th_pcN 0.878;			# post-capping rot capacity for neg loading

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

set th_pP 0.086;		# plastic rot capacity for pos loading
set th_pN 0.086;		# plastic rot capacity for neg loading
set th_pcP 0.598;			# post-capping rot capacity for pos loading
set th_pcN 0.598;			# post-capping rot capacity for neg loading

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

######################## Story 7 - Columns

# col springs @ bottom of Story 7 (above Floor 7) (USE properties from column below)
rotSpring2DModIKModel 3171 17 177 $Ks_col_6 $b $b $Mycol_56 [expr -$Mycol_56] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 3271 27 277 $Ks_col_6 $b $b $Mycol_56 [expr -$Mycol_56] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

set LS 1000;			# basic strength deterioration (a very large # = no cyclic deterioration)
set LK 1000;			# unloading stiffness deterioration (a very large # = no cyclic deterioration)
set LA 1000;			# accelerated reloading stiffness deterioration (a very large # = no cyclic deterioration)
set LD 1000;			# post-capping strength deterioration (a very large # = no deterioration)

set th_pP 0.078;		# plastic rot capacity for pos loading
set th_pN 0.078;		# plastic rot capacity for neg loading
set th_pcP 0.379;			# post-capping rot capacity for pos loading
set th_pcN 0.379;			# post-capping rot capacity for neg loading

# recompute strain hardening for Story 7
set a_mem [expr ($n+1.0)*($Mycol_78*($McMy-1.0)) / ($Ks_col_7*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: there is mistake in Eqn B.5)
#col springs @ top of Story 5 (below Floor 6)
rotSpring2DModIKModel 3172 18 186 $Ks_col_7 $b $b $Mycol_78 [expr -$Mycol_78] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 3272 28 286 $Ks_col_7 $b $b $Mycol_78 [expr -$Mycol_78] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

# recompute strain hardening for Story 8
set a_mem [expr ($n+1.0)*($Mycol_78*($McMy-1.0)) / ($Ks_col_8*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: there is mistake in Eqn B.5)
# col springs @ bottom of Story 8 (above Floor 8)
rotSpring2DModIKModel 3181 18 187 $Ks_col_8 $b $b $Mycol_78 [expr -$Mycol_78] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 3281 28 287 $Ks_col_8 $b $b $Mycol_78 [expr -$Mycol_78] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
#col springs @ top of Story 8 (below Floor 9)
rotSpring2DModIKModel 3182 19 196 $Ks_col_8 $b $b $Mycol_78 [expr -$Mycol_78] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 3282 29 296 $Ks_col_8 $b $b $Mycol_78 [expr -$Mycol_78] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

######################## Story 9 - Columns

# col springs @ bottom of Story 9 (above Floor 9) (USE properties from column below)
rotSpring2DModIKModel 3191 19 197 $Ks_col_8 $b $b $Mycol_78 [expr -$Mycol_78] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 3291 29 297 $Ks_col_8 $b $b $Mycol_78 [expr -$Mycol_78] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

set LS 1000;			# basic strength deterioration (a very large # = no cyclic deterioration)
set LK 1000;			# unloading stiffness deterioration (a very large # = no cyclic deterioration)
set LA 1000;			# accelerated reloading stiffness deterioration (a very large # = no cyclic deterioration)
set LD 1000;			# post-capping strength deterioration (a very large # = no deterioration)

set th_pP 0.063;		# plastic rot capacity for pos loading
set th_pN 0.063;		# plastic rot capacity for neg loading
set th_pcP 0.179;			# post-capping rot capacity for pos loading
set th_pcN 0.179;			# post-capping rot capacity for neg loading

# recompute strain hardening for Story 9
set a_mem [expr ($n+1.0)*($Mycol_910*($McMy-1.0)) / ($Ks_col_9*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: there is mistake in Eqn B.5)
#col springs @ top of Story 9 (below Floor 10)
rotSpring2DModIKModel 3192 110 1106 $Ks_col_9 $b $b $Mycol_910 [expr -$Mycol_910] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 3292 210 2106 $Ks_col_9 $b $b $Mycol_910 [expr -$Mycol_910] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

# recompute strain hardening for Story 10
set a_mem [expr ($n+1.0)*($Mycol_910*($McMy-1.0)) / ($Ks_col_10*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: there is mistake in Eqn B.5)
# col springs @ bottom of Story 10 (above Floor 10)
rotSpring2DModIKModel 31101 110 1107 $Ks_col_10 $b $b $Mycol_910 [expr -$Mycol_910] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 32101 210 2107 $Ks_col_10 $b $b $Mycol_910 [expr -$Mycol_910] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
#col springs @ top of Story 10 (below Floor 11)
rotSpring2DModIKModel 31102 111 1116 $Ks_col_10 $b $b $Mycol_910 [expr -$Mycol_910] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 32102 211 2116 $Ks_col_10 $b $b $Mycol_910 [expr -$Mycol_910] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

######################## Story 11 - Columns

# col springs @ bottom of Story 11 (above Floor 11) (USE properties from column below)
rotSpring2DModIKModel 31111 111 1117 $Ks_col_10 $b $b $Mycol_910 [expr -$Mycol_910] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 32111 211 2117 $Ks_col_10 $b $b $Mycol_910 [expr -$Mycol_910] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

set LS 1000;			# basic strength deterioration (a very large # = no cyclic deterioration)
set LK 1000;			# unloading stiffness deterioration (a very large # = no cyclic deterioration)
set LA 1000;			# accelerated reloading stiffness deterioration (a very large # = no cyclic deterioration)
set LD 1000;			# post-capping strength deterioration (a very large # = no deterioration)

set th_pP 0.072;		# plastic rot capacity for pos loading
set th_pN 0.072;		# plastic rot capacity for neg loading
set th_pcP 0.207;			# post-capping rot capacity for pos loading
set th_pcN 0.207;			# post-capping rot capacity for neg loading

# recompute strain hardening for Story 11
set a_mem [expr ($n+1.0)*($Mycol_1112*($McMy-1.0)) / ($Ks_col_11*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: there is mistake in Eqn B.5)
#col springs @ top of Story 11 (below Floor 12)
rotSpring2DModIKModel 31112 112 1126 $Ks_col_11 $b $b $Mycol_1112 [expr -$Mycol_1112] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 32112 212 2126 $Ks_col_11 $b $b $Mycol_1112 [expr -$Mycol_1112] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

# recompute strain hardening for Story 12
set a_mem [expr ($n+1.0)*($Mycol_1112*($McMy-1.0)) / ($Ks_col_12*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: there is mistake in Eqn B.5)
# col springs @ bottom of Story 12 (above Floor 12)
rotSpring2DModIKModel 31121 112 1127 $Ks_col_12 $b $b $Mycol_1112 [expr -$Mycol_1112] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 32121 212 2127 $Ks_col_12 $b $b $Mycol_1112 [expr -$Mycol_1112] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
#col springs @ top of Story 12 (below Floor 13)
rotSpring2DModIKModel 31122 113 1136 $Ks_col_12 $b $b $Mycol_1112 [expr -$Mycol_1112] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 32122 213 2136 $Ks_col_12 $b $b $Mycol_1112 [expr -$Mycol_1112] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;


# create region for frame column springs
# command: region $regionID -ele $ele_1_ID $ele_2_ID...
region 1 -ele 3111 3211 3112 3212 3121 3221 3122 3222 3131 3231 3132 3232 3141 3241 3142 3242 3151 3251 3152 3252 3161 3261 3162 3262 3171 3271 3172 3272 3181 3281 3182 3282 3191 3291 3192 3292 31101 32101 31102 32102 31111 32111 31112 32112 31121 32121 31122 32122;

#############################################
# define beam springs
# Spring ID: "4xya", where 4 = beam spring, x = Bay #, y = Floor #, a = location in bay
# "a" convention: 1 = left end, 2 = right end
# redefine the rotations since they are not the same

set th_pP 0.039;
set th_pN 0.039;
set th_pcP 0.13;
set th_pcN 0.13;

# Floor 2
set a_mem [expr ($n+1.0)*($Mybeam_2*($McMy-1.0)) / ($Ks_beam_2*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];								# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: there is mistake in Eqn B.5)
#beam springs at Floor 2
####### Switch to beam hinges on the 2/4/6/8/10/12 stories where no braces are connected
equalDOF 12 122 1 2
equalDOF 22 223 1 2

set th_pP 0.047;
set th_pN 0.047;
set th_pcP 0.24;
set th_pcN 0.24;

# Floor 3
set a_mem [expr ($n+1.0)*($Mybeam_3*($McMy-1.0)) / ($Ks_beam_3*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];
#beam springs at Floor 3
rotSpring2DModIKModel 4131 13 132 $Ks_beam_3 $b $b $Mybeam_3 [expr -$Mybeam_3] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 4132 23 233 $Ks_beam_3 $b $b $Mybeam_3 [expr -$Mybeam_3] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

set th_pP 0.039;
set th_pN 0.039;
set th_pcP 0.13;
set th_pcN 0.13;

# Floor 4
set a_mem [expr ($n+1.0)*($Mybeam_4*($McMy-1.0)) / ($Ks_beam_4*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];
#beam springs at Floor 4
####### Switch to beam hinges on the 2/4/6/8/10/12 stories where no braces are connected
equalDOF 14 142 1 2
equalDOF 24 243 1 2

set th_pP 0.047;
set th_pN 0.047;
set th_pcP 0.24;
set th_pcN 0.24;

# Floor 5
set a_mem [expr ($n+1.0)*($Mybeam_5*($McMy-1.0)) / ($Ks_beam_5*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];
#beam springs at Floor 5
rotSpring2DModIKModel 4151 15 152 $Ks_beam_5 $b $b $Mybeam_5 [expr -$Mybeam_5] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 4152 25 253 $Ks_beam_5 $b $b $Mybeam_5 [expr -$Mybeam_5] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

set th_pP 0.039;
set th_pN 0.039;
set th_pcP 0.13;
set th_pcN 0.13;

# Floor 6
set a_mem [expr ($n+1.0)*($Mybeam_6*($McMy-1.0)) / ($Ks_beam_6*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];
#beam springs at Floor 6
####### Switch to beam hinges on the 2/4/6/8/10/12 stories where no braces are connected
equalDOF 16 162 1 2
equalDOF 26 263 1 2

set th_pP 0.045;
set th_pN 0.045;
set th_pcP 0.21;
set th_pcN 0.21;

# Floor 7
set a_mem [expr ($n+1.0)*($Mybeam_7*($McMy-1.0)) / ($Ks_beam_7*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];
#beam springs at Floor 7
rotSpring2DModIKModel 4171 17 172 $Ks_beam_7 $b $b $Mybeam_7 [expr -$Mybeam_7] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 4172 27 273 $Ks_beam_7 $b $b $Mybeam_7 [expr -$Mybeam_7] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

set th_pP 0.039;
set th_pN 0.039;
set th_pcP 0.13;
set th_pcN 0.13;

# Floor 8
set a_mem [expr ($n+1.0)*($Mybeam_8*($McMy-1.0)) / ($Ks_beam_8*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];
#beam springs at Floor 8
####### Switch to beam hinges on the 2/4/6/8/10/12 stories where no braces are connected
equalDOF 18 182 1 2
equalDOF 28 283 1 2

set th_pP 0.045;
set th_pN 0.045;
set th_pcP 0.21;
set th_pcN 0.21;

# Floor 9
set a_mem [expr ($n+1.0)*($Mybeam_9*($McMy-1.0)) / ($Ks_beam_9*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];
#beam springs at Floor 9
rotSpring2DModIKModel 4191 19 192 $Ks_beam_9 $b $b $Mybeam_9 [expr -$Mybeam_9] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 4192 29 293 $Ks_beam_9 $b $b $Mybeam_9 [expr -$Mybeam_9] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

set th_pP 0.039;
set th_pN 0.039;
set th_pcP 0.13;
set th_pcN 0.13;

# Floor 10
set a_mem [expr ($n+1.0)*($Mybeam_10*($McMy-1.0)) / ($Ks_beam_10*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];
#beam springs at Floor 11
####### Switch to beam hinges on the 2/4/6/8/10/12 stories where no braces are connected
equalDOF 110 1102 1 2
equalDOF 210 2103 1 2

set th_pP 0.044;
set th_pN 0.044;
set th_pcP 0.19;
set th_pcN 0.19;

# Floor 11
set a_mem [expr ($n+1.0)*($Mybeam_11*($McMy-1.0)) / ($Ks_beam_11*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];
#beam springs at Floor 11
rotSpring2DModIKModel 41111 111 1112 $Ks_beam_11 $b $b $Mybeam_11 [expr -$Mybeam_11] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 41112 211 2113 $Ks_beam_11 $b $b $Mybeam_11 [expr -$Mybeam_11] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

set th_pP 0.039;
set th_pN 0.039;
set th_pcP 0.13;
set th_pcN 0.13;

# Floor 12
set a_mem [expr ($n+1.0)*($Mybeam_10*($McMy-1.0)) / ($Ks_beam_10*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];
#beam springs at Floor 12
####### Switch to beam hinges on the 2/4/6/8/10/12 stories where no braces are connected
equalDOF 112 1122 1 2
equalDOF 212 2123 1 2

set th_pP 0.042;
set th_pN 0.042;
set th_pcP 0.17;
set th_pcN 0.17;

# Floor 13
set a_mem [expr ($n+1.0)*($Mybeam_13*($McMy-1.0)) / ($Ks_beam_13*$th_pP)];	# strain hardening ratio of spring
set b [expr ($a_mem)/(1.0+$n*(1.0-$a_mem))];
#beam springs at Floor 13
rotSpring2DModIKModel 41131 113 1132 $Ks_beam_13 $b $b $Mybeam_13 [expr -$Mybeam_13] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
rotSpring2DModIKModel 41132 213 2133 $Ks_beam_13 $b $b $Mybeam_13 [expr -$Mybeam_13] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;


# create region for beam springs
region 2 -ele 4152 4151 41132 4171 4172 41131 4131 4132 4192 41111 41112 4191

#########################################################################
#########################################################################
#######Victors Addition - Braces

uniaxialMaterial Steel02 2 58.8 29000.0 0.003 20 0.925 0.15 0.0005 0.01 0.0005 0.01

set mb -0.458
set E0b 0.151
uniaxialMaterial Fatigue 200 2  -E0 $E0b -m $mb

####### Create Fatigue Material #######

##### Connections #####
## uniaxialMaterial Fatigue $matTag $tag <-E0 $E0> <-m $m> <-min $min> <-max $max>
uniaxialMaterial Steel02 3 5880.00 298000.0 0.003 20 0.925 0.15 0.0005 0.01 0.0005 0.01

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
set num_stories 12

# Define the control nodes used to compute story drifts
set ctrl_nodes {
    317
    326
    336
    346
    356
    366
    376
    386
    396
    3106
    3116
    3126
    3136
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

load 18 0.0 [expr {-0.5*(1.05*$Floor8FrameD + 0.25*$Floor8FrameL)}] 0.0
load 28 0.0 [expr {-0.5*(1.05*$Floor8FrameD + 0.25*$Floor8FrameL)}] 0.0
load 386 0.0 [expr {-(1.05*$Floor8LeanD + 0.25*$Floor8LeanL)}] 0.0

load 19 0.0 [expr {-0.5*(1.05*$Floor9FrameD + 0.25*$Floor9FrameL)}] 0.0
load 29 0.0 [expr {-0.5*(1.05*$Floor9FrameD + 0.25*$Floor9FrameL)}] 0.0
load 396 0.0 [expr {-(1.05*$Floor9LeanD + 0.25*$Floor9LeanL)}] 0.0

load 110 0.0 [expr {-0.5*(1.05*$Floor10FrameD + 0.25*$Floor10FrameL)}] 0.0
load 210 0.0 [expr {-0.5*(1.05*$Floor10FrameD + 0.25*$Floor10FrameL)}] 0.0
load 3106 0.0 [expr {-(1.05*$Floor10LeanD + 0.25*$Floor10LeanL)}] 0.0

load 111 0.0 [expr {-0.5*(1.05*$Floor11FrameD + 0.25*$Floor11FrameL)}] 0.0
load 211 0.0 [expr {-0.5*(1.05*$Floor11FrameD + 0.25*$Floor11FrameL)}] 0.0
load 3116 0.0 [expr {-(1.05*$Floor11LeanD + 0.25*$Floor11LeanL)}] 0.0

load 112 0.0 [expr {-0.5*(1.05*$Floor12FrameD + 0.25*$Floor12FrameL)}] 0.0
load 212 0.0 [expr {-0.5*(1.05*$Floor12FrameD + 0.25*$Floor12FrameL)}] 0.0
load 3126 0.0 [expr {-(1.05*$Floor12LeanD + 0.25*$Floor12LeanL)}] 0.0

load 113 0.0 [expr {-0.5*(1.05*$Floor13FrameD + 0.25*$Floor13FrameL)}] 0.0
load 213 0.0 [expr {-0.5*(1.05*$Floor13FrameD + 0.25*$Floor13FrameL)}] 0.0
load 3136 0.0 [expr {-(1.05*$Floor13LeanD + 0.25*$Floor13LeanL)}] 0.0

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
region 4 -eleRange 111 2122 rayleigh 0.0 0.0 $a1_mod 0.0;	# assign stiffness proportional damping to frame beams & columns w/ n modifications
rayleigh $a0 0.0 0.0 0.0;          				# assign mass proportional damping to structure (only assigns to nodes with mass)


region 5 -eleRange 1122100 3928700 rayleigh 0.0 0.0 $a1 0.0;	# assign stiffness proportional damping to frame braces
region 6 -eleRange 19210100 39210700 rayleigh 0.0 0.0 $a1 0.0;	# assign stiffness proportional damping to frame braces
region 7 -eleRange 111210100 113212700 rayleigh 0.0 0.0 $a1 0.0;	# assign stiffness proportional damping to frame braces
region 8 -eleRange 311210100 313212700 rayleigh 0.0 0.0 $a1 0.0;	# assign stiffness proportional damping to frame braces
#--------------------------------------------------------------------------------------------------------------------------------------------------------



