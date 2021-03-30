#### Round HSS Sections

#### patch circ $matTag $numSubdivCirc $numSubdivRad $yCenter $zCenter $intRad $extRad <$startAng $endAng>

set numSubdivCirc 12
set numSubdivRad 3

######################### STORY 1 ###############################
set sec_used1 11;
#HSS 9.625 x 0.5
set OD 9.625;		#Outer Diameter
set ID 8.695;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 11 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}


######################### STORY 2 ###############################
set sec_used2 12;
#HSS 8.75 x 0.5
set OD 8.75;		#Outer Diameter
set ID 7.82;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 12 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}

######################### STORY 3 ###############################
set sec_used3 13;
#HSS 8.75 x .312
set OD 8.75;		#Outer Diameter
set ID 8.168;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 13 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}


