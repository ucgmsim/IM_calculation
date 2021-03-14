#### Round HSS Sections

#### patch circ $matTag $numSubdivCirc $numSubdivRad $yCenter $zCenter $intRad $extRad <$startAng $endAng>

set numSubdivCirc 10
set numSubdivRad 6

######################### STORY 1 ###############################
set sec_used1 11;
#HSS 9-5/8 x 0.5
set OD 9.625;		#Outer Diameter
set ID 8.695;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 11 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}


######################### STORY 2 ###############################
set sec_used2 12;
#HSS 9-5/8 x 0.5
set OD 9.625;		#Outer Diameter
set ID 8.695;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 12 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}


######################### STORY 3 ###############################
set sec_used3 13;
#HSS 9-5/8 x 0.5
set OD 9.625;		#Outer Diameter
set ID 8.695;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 13 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}


######################### STORY 4 ###############################
set sec_used4 14;
#HSS 9-5/8 x 0.5
set OD 9.625;		#Outer Diameter
set ID 8.695;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 14 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}


######################### STORY 5 ###############################
set sec_used5 15;
#HSS 10 x 0.375
set OD 10;		#Outer Diameter
set ID 9.302;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 15 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}


######################### STORY 6 ###############################
set sec_used6 16;
#HSS 10 x 0.375
set OD 10;		#Outer Diameter
set ID 9.302;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 16 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}


######################### STORY 7 ###############################
set sec_used7 17;
#HSS 10 x 0.375
set OD 10;		#Outer Diameter
set ID 9.302;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 17 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}

######################### STORY 8 ###############################
set sec_used8 18;
#HSS 10 x 0.375
set OD 10;		#Outer Diameter
set ID 9.302;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 18 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}

######################### STORY 9 ###############################
set sec_used9 19;
#HSS 8-3/4 x 0.312
set OD 8.75;		#Outer Diameter
set ID 8.168;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 19 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}

######################### STORY 10 ###############################
set sec_used10 110;
#HSS 8-3/4 x 0.312
set OD 8.75;		#Outer Diameter
set ID 8.168;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 110 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}

######################### STORY 11 ###############################
set sec_used11 111;
#HSS 6-5/8 x 0.312
set OD 6;		#Outer Diameter
set ID 5.418;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 111 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}

######################### STORY 12 ###############################
set sec_used12 112;
#HSS 6-5/8 x 0.312
set OD 6;		#Outer Diameter
set ID 5.418;		#Inner Diameter
set OR [expr $OD/2];	#Outer Radius
set IR [expr $ID/2];	#Inner Radius

section fiberSec 112 {patch circ 200 $numSubdivCirc $numSubdivRad 0.0 0.0 $IR $OR 0 360}




