# Set the brace mass inflation factor to control the maximum permissible dt_analysis
# Chosen such that a relatively high dt_analysis is obtained without increasing the
# fundamental mode period too much. Care is also taken to ensure mass in braces is
# small in comparison to mass at story level.
set mass_inf_factor 50.0

##############################################################################################
####### Define Brace Notes

set ImpMag 1000;   #Imperfection is = L/ImpMag

set rise $HStory1
set run  [expr $WBay/2]
set angle [expr atan($rise/$run)]
set efflength 0.7;

set Lptp [expr pow((pow($rise,2)+pow($run,2)),0.5)];	#Center to center brace length
set Lb [expr $Lptp*$efflength]; 			#Effective brace length
set stiff [expr ($Lptp - $Lb)/2];			#Length of stiff section at the end of braces


set length [expr $Lb*cos($angle)]
set height [expr $Lb*sin($angle)]

set impX [expr ($Lb/$ImpMag)*sin($angle)]
set impY [expr ($Lb/$ImpMag)*cos($angle)]

set Lh_GP 2;  #Gusset Plate Length / Zero-Stiffness Spring

##############################################################################################
# Nodes go from down left corner to up right corner


# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX 1;
set dirY 1;
set ampX -1;
set ampY 1;

set x [expr $Pier1  + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor1 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 1001	$x	$y
node 1002	$x	$y
node 1003 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 100301 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 100302 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 100303 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 100304 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 100305 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 1004 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 100401 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 100402 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 100403 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 100404 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 100405 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 1005 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 1006 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 1001 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 1002 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 1003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 100301 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 100302 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 100303 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 100304 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 100305 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 100401 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 100402 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 100403 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 100404 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 100405 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1006 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################
# Nodes go from down right corner to up left corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX -1;
set dirY 1;
set ampX 1;
set ampY 1;

set x [expr $Pier2  + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor1 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 1007	$x	$y
node 1008	$x	$y
node 1009 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 100901 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 100902 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 100903 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 100904 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 100905 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 1010 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 101001 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 101002 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 101003 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 101004 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 101005 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 1011 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 1012 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 1007 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 1008 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 1009 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 100901 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 100902 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 100903 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 100904 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 100905 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1010 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 101001 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 101002 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 101003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 101004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 101005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1011 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1012 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################
# Nodes go from up left corner to down right corner

set rise [expr $HStoryTyp]
set run  [expr $WBay/2]
set angle [expr atan($rise/$run)]
set efflength 0.7;

set Lptp [expr pow((pow($rise,2)+pow($run,2)),0.5)];	#Center to center brace length
set Lb [expr $Lptp*$efflength]; 			#Effective brace length
set stiff [expr ($Lptp - $Lb)/2];			#Length of stiff section at the end of braces

set length [expr $Lb*cos($angle)]
set height [expr $Lb*sin($angle)]

set impX [expr ($Lb/$ImpMag)*sin($angle)]
set impY [expr ($Lb/$ImpMag)*cos($angle)]


# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX 1;
set dirY -1;
set ampX 1;
set ampY 1;

set x [expr $Pier1 + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor3 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 2001	$x	$y
node 2002	$x	$y
node 2003 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 200301 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 200302 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 200303 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 200304 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 200305 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 2004 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 200401 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 200402 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 200403 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 200404 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 200405 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 2005 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 2006 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 2001 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 2002 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 2003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 200301 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 200302 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 200303 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 200304 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 200305 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 2004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 200401 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 200402 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 200403 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 200404 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 200405 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 2005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 2006 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################
# Nodes go from up right corner to down left corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX -1;
set dirY -1;
set ampX -1;
set ampY 1;

set x [expr $Pier2  + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor3 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 2007	$x	$y
node 2008	$x	$y
node 2009 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 200901 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 200902 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 200903 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 200904 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 200905 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 2010 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 201001 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 201002 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 201003 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 201004 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 201005 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 2011 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 2012 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 2007 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 2008 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 2009 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 200901 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 200902 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 200903 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 200904 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 200905 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 2010 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 201001 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 201002 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 201003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 201004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 201005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 2011 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 2012 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################

##############################################################################################
# Nodes go from down left corner to up right corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX 1;
set dirY 1;
set ampX -1;
set ampY 1;

set x [expr $Pier1 + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor3 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 3001	$x	$y
node 3002	$x	$y
node 3003 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 300301 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 300302 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 300303 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 300304 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 300305 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 3004 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 300401 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 300402 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 300403 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 300404 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 300405 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 3005 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 3006 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 3001 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 3002 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 3003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 300301 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 300302 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 300303 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 300304 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 300305 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 3004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 300401 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 300402 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 300403 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 300404 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 300405 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 3005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 3006 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################
# Nodes go from down right corner to up left corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX -1;
set dirY 1;
set ampX 1;
set ampY 1;

set x [expr $Pier2  + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor3 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 3007	$x	$y
node 3008	$x	$y
node 3009 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 300901 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 300902 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 300903 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 300904 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 300905 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 3010 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 301001 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 301002 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 301003 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 301004 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 301005 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 3011 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 3012 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 3007 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 3008 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 3009 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 300901 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 300902 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 300903 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 300904 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 300905 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 3010 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 301001 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 301002 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 301003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 301004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 301005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 3011 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 3012 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################

##############################################################################################
# Nodes go from up left corner to down right corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX 1;
set dirY -1;
set ampX 1;
set ampY 1;

set x [expr $Pier1 + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor5 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 4001	$x	$y
node 4002	$x	$y
node 4003 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 400301 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 400302 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 400303 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 400304 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 400305 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 4004 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 400401 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 400402 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 400403 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 400404 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 400405 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 4005 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 4006 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 4001 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 4002 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 4003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 400301 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 400302 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 400303 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 400304 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 400305 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 4004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 400401 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 400402 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 400403 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 400404 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 400405 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 4005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 4006 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################
# Nodes go from up right corner to down left corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX -1;
set dirY -1;
set ampX -1;
set ampY 1;

set x [expr $Pier2  + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor5 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 4007	$x	$y
node 4008	$x	$y
node 4009 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 400901 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 400902 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 400903 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 400904 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 400905 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 4010 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 401001 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 401002 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 401003 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 401004 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 401005 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 4011 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 4012 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 4007 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 4008 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 4009 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 400901 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 400902 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 400903 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 400904 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 400905 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 4010 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 401001 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 401002 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 401003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 401004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 401005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 4011 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 4012 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################

##############################################################################################
# Nodes go from down left corner to up right corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX 1;
set dirY 1;
set ampX -1;
set ampY 1;

set x [expr $Pier1 + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor5 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 5001	$x	$y
node 5002	$x	$y
node 5003 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 500301 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 500302 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 500303 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 500304 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 500305 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 5004 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 500401 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 500402 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 500403 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 500404 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 500405 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 5005 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 5006 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 5001 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 5002 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 5003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 500301 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 500302 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 500303 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 500304 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 500305 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 5004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 500401 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 500402 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 500403 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 500404 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 500405 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 5005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 5006 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################
# Nodes go from down right corner to up left corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX -1;
set dirY 1;
set ampX 1;
set ampY 1;

set x [expr $Pier2  + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor5 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 5007	$x	$y
node 5008	$x	$y
node 5009 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 500901 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 500902 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 500903 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 500904 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 500905 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 5010 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 501001 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 501002 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 501003 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 501004 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 501005 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 5011 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 5012 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 5007 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 5008 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 5009 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 500901 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 500902 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 500903 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 500904 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 500905 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 5010 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 501001 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 501002 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 501003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 501004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 501005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 5011 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 5012 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################

##############################################################################################
# Nodes go from up left corner to down right corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX 1;
set dirY -1;
set ampX 1;
set ampY 1;

set x [expr $Pier1 + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor7 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 6001	$x	$y
node 6002	$x	$y
node 6003 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 600301 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 600302 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 600303 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 600304 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 600305 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 6004 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 600401 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 600402 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 600403 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 600404 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 600405 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 6005 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 6006 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 6001 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 6002 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 6003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 600301 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 600302 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 600303 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 600304 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 600305 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 6004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 600401 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 600402 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 600403 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 600404 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 600405 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 6005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 6006 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################
# Nodes go from up right corner to down left corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX -1;
set dirY -1;
set ampX -1;
set ampY 1;

set x [expr $Pier2  + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor7 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 6007	$x	$y
node 6008	$x	$y
node 6009 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 600901 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 600902 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 600903 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 600904 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 600905 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 6010 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 601001 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 601002 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 601003 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 601004 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 601005 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 6011 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 6012 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 6007 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 6008 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 6009 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 600901 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 600902 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 600903 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 600904 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 600905 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 6010 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 601001 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 601002 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 601003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 601004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 601005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 6011 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 6012 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################

##############################################################################################
# Nodes go from down left corner to up right corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX 1;
set dirY 1;
set ampX -1;
set ampY 1;

set x [expr $Pier1 + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor7 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 7001	$x	$y
node 7002	$x	$y
node 7003 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 700301 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 700302 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 700303 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 700304 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 700305 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 7004 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 700401 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 700402 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 700403 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 700404 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 700405 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 7005 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 7006 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 7001 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 7002 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 7003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 700301 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 700302 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 700303 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 700304 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 700305 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 7004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 700401 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 700402 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 700403 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 700404 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 700405 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 7005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 7006 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################
# Nodes go from down right corner to up left corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX -1;
set dirY 1;
set ampX 1;
set ampY 1;

set x [expr $Pier2  + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor7 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 7007	$x	$y
node 7008	$x	$y
node 7009 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 700901 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 700902 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 700903 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 700904 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 700905 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 7010 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 701001 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 701002 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 701003 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 701004 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 701005 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 7011 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 7012 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 7007 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 7008 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 7009 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 700901 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 700902 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 700903 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 700904 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 700905 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 7010 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 701001 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 701002 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 701003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 701004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 701005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 7011 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 7012 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################

##############################################################################################
# Nodes go from up left corner to down right corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX 1;
set dirY -1;
set ampX 1;
set ampY 1;

set x [expr $Pier1 + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor9 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 8001	$x	$y
node 8002	$x	$y
node 8003 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 800301 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 800302 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 800303 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 800304 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 800305 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 8004 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 800401 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 800402 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 800403 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 800404 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 800405 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 8005 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 8006 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 8001 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 8002 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 8003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 800301 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 800302 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 800303 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 800304 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 800305 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 8004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 800401 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 800402 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 800403 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 800404 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 800405 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 8005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 8006 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################
# Nodes go from up right corner to down left corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX -1;
set dirY -1;
set ampX -1;
set ampY 1;

set x [expr $Pier2  + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor9 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 8007	$x	$y
node 8008	$x	$y
node 8009 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 800901 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 800902 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 800903 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 800904 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 800905 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 8010 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 801001 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 801002 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 801003 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 801004 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 801005 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 8011 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 8012 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 8007 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 8008 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 8009 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 800901 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 800902 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 800903 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 800904 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 800905 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 8010 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 801001 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 801002 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 801003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 801004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 801005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 8011 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 8012 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]


##############################################################################################

##############################################################################################
# Nodes go from down left corner to up right corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX 1;
set dirY 1;
set ampX -1;
set ampY 1;

set x [expr $Pier1 + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor9 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 9001	$x	$y
node 9002	$x	$y
node 9003 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 900301 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 900302 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 900303 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 900304 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 900305 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 9004 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 900401 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 900402 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 900403 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 900404 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 900405 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 9005 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 9006 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 9001 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 9002 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 9003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 900301 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 900302 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 900303 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 900304 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 900305 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 9004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 900401 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 900402 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 900403 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 900404 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 900405 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 9005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 9006 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################
# Nodes go from down right corner to up left corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX -1;
set dirY 1;
set ampX 1;
set ampY 1;

set x [expr $Pier2  + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor9 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 9007	$x	$y
node 9008	$x	$y
node 9009 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 900901 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 900902 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 900903 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 900904 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 900905 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 9010 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 901001 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 901002 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 901003 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 901004 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 901005 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 9011 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 9012 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 9007 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 9008 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 9009 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 900901 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 900902 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 900903 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 900904 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 900905 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 9010 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 901001 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 901002 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 901003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 901004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 901005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 9011 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 9012 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################

##############################################################################################
# Nodes go from up left corner to down right corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX 1;
set dirY -1;
set ampX 1;
set ampY 1;

set x [expr $Pier1 + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor11 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 10001	$x	$y
node 10002	$x	$y
node 10003 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 1000301 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 1000302 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 1000303 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 1000304 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 1000305 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 10004 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 1000401 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 1000402 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 1000403 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 1000404 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 1000405 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 10005 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 10006 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 10001 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 10002 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 10003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1000301 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1000302 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1000303 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1000304 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1000305 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 10004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1000401 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1000402 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1000403 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1000404 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1000405 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 10005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 10006 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################
# Nodes go from up right corner to down left corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX -1;
set dirY -1;
set ampX -1;
set ampY 1;

set x [expr $Pier2  + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor11 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 10007	$x	$y
node 10008	$x	$y
node 10009 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 1000901 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 1000902 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 1000903 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 1000904 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 1000905 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 10010 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 1001001 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 1001002 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 1001003 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 1001004 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 1001005 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 10011 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 10012 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 10007 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 10008 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 10009 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1000901 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1000902 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1000903 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1000904 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1000905 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 10010 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1001001 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1001002 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1001003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1001004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1001005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 10011 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 10012 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]


##############################################################################################

##############################################################################################
# Nodes go from down left corner to up right corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX 1;
set dirY 1;
set ampX -1;
set ampY 1;

set x [expr $Pier1 + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor11 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 11001	$x	$y
node 11002	$x	$y
node 11003 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 1100301 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 1100302 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 1100303 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 1100304 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 1100305 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 11004 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 1100401 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 1100402 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 1100403 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 1100404 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 1100405 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 11005 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 11006 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 11001 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 11002 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 11003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1100301 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1100302 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1100303 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1100304 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1100305 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 11004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1100401 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1100402 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1100403 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1100404 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1100405 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 11005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 11006 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################
# Nodes go from down right corner to up left corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX -1;
set dirY 1;
set ampX 1;
set ampY 1;

set x [expr $Pier2  + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor11 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 11007	$x	$y
node 11008	$x	$y
node 11009 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 1100901 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 1100902 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 1100903 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 1100904 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 1100905 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 11010 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 1101001 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 1101002 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 1101003 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 1101004 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 1101005 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 11011 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 11012 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 11007 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 11008 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 11009 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1100901 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1100902 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1100903 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1100904 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1100905 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 11010 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1101001 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1101002 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1101003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1101004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1101005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 11011 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 11012 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################

##############################################################################################
# Nodes go from up left corner to down right corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX 1;
set dirY -1;
set ampX 1;
set ampY 1;

set x [expr $Pier1 + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor13 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 12001	$x	$y
node 12002	$x	$y
node 12003 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 1200301 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 1200302 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 1200303 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 1200304 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 1200305 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 12004 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 1200401 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 1200402 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 1200403 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 1200404 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 1200405 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 12005 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 12006 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 12001 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 12002 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 12003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1200301 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1200302 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1200303 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1200304 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1200305 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 12004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1200401 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1200402 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1200403 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1200404 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1200405 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 12005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 12006 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]



##############################################################################################
# Nodes go from up right corner to down left corner

# Assuming brace weight of 50lb/ft
set BraceWt [expr {50.0*$mass_inf_factor}]
set BraceMass [expr {$BraceWt/1e3/12.0/$g*$Lptp}]

set dirX -1;
set dirY -1;
set ampX -1;
set ampY 1;

set x [expr $Pier2  + $dirX * $stiff * cos($angle)];	#Adjust starting point to include stiff end length
set y [expr $Floor13 + $dirY * $stiff * sin($angle)];	#Adjust starting point to include stiff end length

node 12007	$x	$y
node 12008	$x	$y
node 12009 [expr $x + $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$Lh_GP*sin($angle)]

node 1200901 [expr $x + $dirX*$length*1/10 + $ampX*$impX*2*1/10] [expr $y + $dirY*$height*1/10 + $ampY*$impY*2*1/10]
node 1200902 [expr $x + $dirX*$length*2/10 + $ampX*$impX*2*2/10] [expr $y + $dirY*$height*2/10 + $ampY*$impY*2*2/10]
node 1200903 [expr $x + $dirX*$length*3/10 + $ampX*$impX*2*3/10] [expr $y + $dirY*$height*3/10 + $ampY*$impY*2*3/10]
node 1200904 [expr $x + $dirX*$length*4/10 + $ampX*$impX*2*4/10] [expr $y + $dirY*$height*4/10 + $ampY*$impY*2*4/10]
node 1200905 [expr $x + $dirX*$length*9/20 + $ampX*$impX*2*9/20] [expr $y + $dirY*$height*9/20 + $ampY*$impY*2*9/20]

node 12010 [expr $x + $dirX*$length*2/4 + $ampX*$impX*1.0] [expr $y + $dirY*$height*2/4 + $ampY*$impY*1.0]

node 1201001 [expr $x + $dirX*$length*11/20 + $ampX*$impX*2*11/20] [expr $y + $dirY*$height*11/20 + $ampY*$impY*2*11/20]
node 1201002 [expr $x + $dirX*$length*6/10 + $ampX*$impX*2*6/10] [expr $y + $dirY*$height*6/10 + $ampY*$impY*2*6/10]
node 1201003 [expr $x + $dirX*$length*7/10 + $ampX*$impX*2*7/10] [expr $y + $dirY*$height*7/10 + $ampY*$impY*2*7/10]
node 1201004 [expr $x + $dirX*$length*8/10 + $ampX*$impX*2*8/10] [expr $y + $dirY*$height*8/10 + $ampY*$impY*2*8/10]
node 1201005 [expr $x + $dirX*$length*9/10 + $ampX*$impX*2*9/10] [expr $y + $dirY*$height*9/10 + $ampY*$impY*2*9/10]

node 12011 [expr $x + $dirX*$length*4/4 - $dirX*$Lh_GP*cos($angle)] [expr $y + $dirY*$height*4/4 - $dirY*$Lh_GP*sin($angle)]
node 12012 [expr $x + $dirX*$length*4/4 + $ampX*$impX*0.0] [expr $y + $dirY*$height*4/4 + $ampY*$impY*0.0]

mass 12007 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 12008 [expr {$BraceMass/30.0}] [expr {$BraceMass/30.0}] [expr {$RotInertia/100.0}]
mass 12009 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1200901 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1200902 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1200903 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1200904 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1200905 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 12010 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1201001 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1201002 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1201003 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1201004 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 1201005 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 12011 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]
mass 12012 [expr {$BraceMass/15.0}] [expr {$BraceMass/15.0}] [expr {$RotInertia/50.0}]


##############################################################################################

