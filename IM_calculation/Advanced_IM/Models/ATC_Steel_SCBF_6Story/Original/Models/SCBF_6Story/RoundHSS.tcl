#### Round HSS Sections

#### patch circ $matTag $numSubdivCirc $numSubdivRad $yCenter $zCenter $intRad $extRad <$startAng $endAng>

set numSubdivCirc 12
set numSubdivRad 3

######################### STORY 1 ###############################
set sec_used1 11;
#HSS 12.5 x 0.5
set OD 12.5;		#Outer Diameter
set ID 11.57;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

### Material Tag 16 is for braces that will fracture at about 0.02 Strain
#section fiberSec 11 {patch circ 2 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}
section fiberSec 11 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}

## Defining a connection that can fracture
#set sec_used1a 111;
#set sec_used1b 112;
#section fiberSec 111 {patch circ 4 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}
#section fiberSec 112 {patch circ 5 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}

######################### STORY 2 ###############################
set sec_used2 12;
#HSS 12.5 x 0.5
set OD 12.5;		#Outer Diameter
set ID 11.57;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 12 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}


######################### STORY 3 ###############################
set sec_used3 13;
#HSS 11.25 x 0.5
set OD 11.25;		#Outer Diameter
set ID 10.32;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 13 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}


######################### STORY 4 ###############################
set sec_used4 14;
#HSS 9.625 x 0.5
set OD 9.625;		#Outer Diameter
set ID 8.692;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 14 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}


######################### STORY 5 ###############################
set sec_used5 15;
#HSS 9.625 x 0.375
set OD 9.625;		#Outer Diameter
set ID 8.927;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 15 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}


######################### STORY 6 ###############################
#HSS 7.5 x 0.312
set sec_used6 16;
set OD 7.50;		#Outer Diameter
set ID 6.918;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 16 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}