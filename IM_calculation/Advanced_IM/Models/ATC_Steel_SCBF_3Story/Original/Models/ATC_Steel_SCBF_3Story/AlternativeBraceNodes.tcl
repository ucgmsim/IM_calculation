# Set the brace mass inflation factor to control the maximum permissible dt_analysis
# Chosen such that a relatively high dt_analysis is obtained without increasing the
# fundamental mode period too much. Care is also taken to ensure mass in braces is
# small in comparison to mass at story level.
set mass_inf_factor 5.0

##############################################################################################
####### Define Brace Nodes

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
#node 2009 [expr $x + $dirX*($Lh_GP+2)*cos($angle)] [expr $y + $dirY*($Lh_GP+2)*sin($angle)]

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
#node 3003 [expr $x + $dirX*($Lh_GP+2)*cos($angle)] [expr $y + $dirY*($Lh_GP+2)*sin($angle)]

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
#node 3009 [expr $x + $dirX*($Lh_GP+2)*cos($angle)] [expr $y + $dirY*($Lh_GP+2)*sin($angle)]

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




