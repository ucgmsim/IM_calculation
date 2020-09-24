#########################################################################################################################
################################### 5-Story Moment Frame , Wenhao Chen, 2012 ############################################
#########################################################################################################################


#########################################################################################################################
### DEFINE MATERIALS

# Define Model Builder
	model BasicBuilder -ndm 2 -ndf 3;								# Define the model builder, ndm = #dimension, ndf = #dofs
	

# Define Variables, Units and Constants
	source [file join [file dirname [info script]] DefineUnitsAndConstants.tcl]
	source [file join [file dirname [info script]] DefineVariables.tcl]

# Source Display Procedures
	# source DisplayModel2D.tcl;			# procedure for displaying a 2D perspective of model
	# source DisplayPlane.tcl;			# procedure for displaying a plane in a model
	
# Build Model
	source [file join [file dirname [info script]] DefineFunctionsAndProcedures.tcl]
#	source SetAnalysisOptions.tcl


##############################################################################################################################
#          				 						  Define Building Geometry Parameters			     					     #                    
##############################################################################################################################

# Define structure-geometry parameters
	set NStories 5.0;														# number of stories
    set NMBays 3.0;														    # number of moment frame bays
	set NGBays 0.0;														    # number of gravity frame bays 
	set NBays [expr $NMBays+$NGBays+2];										# number of frame bays (inlcudesw bay for P-delta column and gravity system)
	set WBay1 [expr 29*12];													# width of bay 1
	set WBay2 [expr 29*12];													# width of bay 2
	set WBay3 [expr 29*12];													# width of bay 3
	set WBay4 [expr 29*12];													# width of bay 4
	set HStory [expr 13*12];												# typical height between beams centerline
	set BeamSlab 6.5;														# Distance between beam centerline and slab top
	set HStory1 [expr $HStory-$BeamSlab];									# clear hieght of fisrt story
	set HBuilding [expr ($NStories-1)*$HStory+$HStory1];					# height of building
 
# Define locations of beam/column joints:
	set Pier1  0.0;					        							# Column Line 1
	set Pier2  [expr $Pier1 + $WBay1];					        		# Column Line 2
	set Pier3  [expr $Pier2 + $WBay2];					        		# Column Line 3
	set Pier4  [expr $Pier3 + $WBay3];					        		# Column Line 4
	set Pier5  [expr $Pier4 + $WBay4];					        		# Column Line 5 : P-Delta Column Line
	set Piermid1 [expr ($Pier2+$Pier3)/2];								# Point Line to Connect to Leaning Column

	set Floor1 0.0;				                						# 1st Floor
	set Floor2 [expr $Floor1 + $HStory1];								# 2nd Floor
	set Floor3 [expr $Floor2 + $HStory];								# 3rd Floor
	set Floor4 [expr $Floor3 + $HStory];								# 4th Floor
	set Floor5 [expr $Floor4 + $HStory];								# 5th Floor
	set Floor6 [expr $Floor5 + $HStory];								# 6th Floor

 # puts "geometry parameters defined"

##############################################################################################################################
#          										   Define Section Prperties										             #
##############################################################################################################################

# define column section W24X103 section
	set Bcol1 9.00;													    # Column Width
	set Hcol1 24.4;														# Column Depth
	set tfcol1 0.98;													# Column flange thickness
	set twcol1 0.55;													# Column web thickness
	set Acol1 30.3;										                # Column cross-sectional area
	set Icol1 3000;					                                    # Column moment of inertia
	set Icol_mod1  [expr $Icol1*($stiffFactor1+1.0)/$stiffFactor1];		# modified moment of inertia for columns accounting for being in series with plastic spring
	set Ks_col_1   [expr $stiffFactor1*6.0*$Es*$Icol_mod1/$HStory];		# rotational stiffness of moment frame column springs
	
# define column section W24X117 section
	set Bcol2 12.8;													    # Column Width
	set Hcol2 24.3;														# Column Depth
	set tfcol2 0.85;													# Column flange thickness
	set twcol2 0.550;													# Column web thickness
	set Acol2 34.4;										                # Column cross-sectional area
	set Icol2 3540;					                                    # Column moment of inertia
	set Icol_mod2  [expr $Icol2*($stiffFactor1+1.0)/$stiffFactor1];		# modified moment of inertia for columns accounting for being in series with plastic spring
	set Ks_col_2   [expr $stiffFactor1*6.0*$Es*$Icol_mod2/$HStory];		# rotational stiffness of moment frame column springs
	
# define column section W24X146 section
	set Bcol3 12.9;													    # Column Width
	set Hcol3 24.7;														# Column Depth
	set tfcol3 1.09;													# Column flange thickness
	set twcol3 0.650;													# Column web thickness
	set Acol3 43.0;										                # Column cross-sectional area
	set Icol3 4580;					                                    # Column moment of inertia
	set Icol_mod3  [expr $Icol3*($stiffFactor1+1.0)/$stiffFactor1];		# modified moment of inertia for columns accounting for being in series with plastic spring
	set Ks_col_3   [expr $stiffFactor1*6.0*$Es*$Icol_mod3/$HStory];		# rotational stiffness of moment frame column springs
	
# define column section W24X176 section
	set Bcol4 12.9;													    # Column Width
	set Hcol4 25.5;														# Column Depth
	set tfcol4 1.34;													# Column flange thickness
	set twcol4 0.750;													# Column web thickness
	set Acol4 51.7;										                # Column cross-sectional area
	set Icol4 5680;					                                    # Column moment of inertia
	set Icol_mod4  [expr $Icol4*($stiffFactor1+1.0)/$stiffFactor1];		# modified moment of inertia for columns accounting for being in series with plastic spring
	set Ks_col_4   [expr $stiffFactor1*6.0*$Es*$Icol_mod4/$HStory];		# rotational stiffness of moment frame column springs
		
# define beam section W18X50 section
	set Bbeam1 7.50;													# Beam Width
	set Hbeam1 18.0;													# Beam Depth
	set Abeam1 14.7;									                # Beam cross-sectional area
	set Ibeam1  800;													# Beam moment of inertia
	set Ibeam_mod1 [expr $Ibeam1*($stiffFactor1+1.0)/$stiffFactor1];		# modified moment of inertia for beams
	set Ks_beam_1 [expr $stiffFactor1*6.0*$Es*$Ibeam_mod1/$WBay1];					# rotational stiffness of moment frame beam springs
	
# define beam section W21X62 section
	set Bbeam2 8.24;													# Beam Width
	set Hbeam2 21.0;													# Beam Depth
	set Abeam2 18.3;									                # Beam cross-sectional area
	set Ibeam2 1330;													# Beam moment of inertia
	set Ibeam_mod2 [expr $Ibeam2*($stiffFactor1+1.0)/$stiffFactor1];		# modified moment of inertia for beams
	set Ks_beam_2 [expr $stiffFactor1*6.0*$Es*$Ibeam_mod2/$WBay1];					# rotational stiffness of moment frame beam springs

# define beam section W21X73 section
	set Bbeam3 8.30;													# Beam Width
	set Hbeam3 21.2;													# Beam Depth
	set Abeam3 21.5;									                # Beam cross-sectional area
	set Ibeam3 1600;													# Beam moment of inertia
	set Ibeam_mod3 [expr $Ibeam3*($stiffFactor1+1.0)/$stiffFactor1];		# modified moment of inertia for beams
	set Ks_beam_3 [expr $stiffFactor1*6.0*$Es*$Ibeam_mod3/$WBay1];					# rotational stiffness of moment frame beam springs
	
# calculate panel zone dimensions
	set pzlat1  [expr ($Hcol1+$Hcol2+$Hcol3+$Hcol4)/4.0/2.0];	# lateral dist from CL of beam-col joint to edge of panel zone (= half the column depth), MF
	set pzlat2  [expr ($Hcol1+$Hcol2+$Hcol3+$Hcol4)/4.0/2.0];	# lateral dist from CL of beam-col joint to edge of panel zone (= half the column depth), MF
	set pzvert1 [expr $Hbeam1/2.0];	# vert dist from CL of beam-col joint to edge of panel zone (= half the beam depth), MF
	set pzvert2 [expr ($Hbeam1+$Hbeam2)/2.0/2.0];	# vert dist from CL of beam-col joint to edge of panel zone (= half the beam depth), MF	
	
	
# calculate plastic hinge offsets from beam-column centerlines:
    set a 4;										# lateral dist from column flange to RBS
	set b 14	; 									# length of RBS
	set phlat1 [expr $pzlat1 + $a + $b/2.0];			# lateral dist from CL of beam-col joint to beam hinge on moment frames
	set phvert1 [expr $pzvert1 + 0.0];				# vert dist from CL of beam-col joint to column hinge (forms at edge of panel zone) on moment frames
    set a 5;										# lateral dist from column flange to RBS
	set b 16	; 									# length of RBS
	set phlat2 [expr $pzlat2 + $a + $b/2.0];			# lateral dist from CL of beam-col joint to beam hinge on moment frames
	set phvert2 [expr $pzvert2 + 0.0];				# vert dist from CL of beam-col joint to column hinge (forms at edge of panel zone) on moment frames
	
# puts "section properties defined"



#########################################################################################################################
#                                                Define Rotational Springs for Panel Zones												  
#########################################################################################################################
	
# Trilinear Spring
# Strai Hardening 
	set as 0.03;
# Yield Shear
	set Vy [expr 0.55 * $fy * $Hcol1 * $twcol1];
# Shear Modulus
	set G [expr $Es/(2.0 * (1.0 + 0.30))]
# Elastic Stiffness
	set Ke [expr 0.95 * $G * $twcol1 * $Hcol1];
# Plastic Stiffness
	set Kp [expr 0.95 * $G * $Bcol1 * ($tfcol1 * $tfcol1) / $Hbeam1];

# Define Trilinear Equivalent Rotational Spring
# Yield point for Trilinear Spring at gamma1_y
	set gamma1_y [expr $Vy/$Ke]; set M1y [expr $gamma1_y * ($Ke * $Hbeam1)];
# Second Point for Trilinear Spring at 4 * gamma1_y
	set gamma2_y [expr 4.0 * $gamma1_y]; set M2y [expr $M1y + ($Kp * $Hbeam1) * ($gamma2_y - $gamma1_y)];
# Third Point for Trilinear Spring at 100 * gamma1_y
	set gamma3_y [expr 100.0 * $gamma1_y]; set M3y [expr $M2y + ($as * $Ke * $Hbeam1) * ($gamma3_y - $gamma2_y)];
    
	uniaxialMaterial Hysteretic $PanelZoneM1 $M1y $gamma1_y  $M2y $gamma2_y $M3y $gamma3_y [expr -$M1y] [expr -$gamma1_y] [expr -$M2y] [expr -$gamma2_y] [expr -$M3y] [expr -$gamma3_y] 1 1 0.0 0.0 0.0

# Trilinear Spring
# Strai Hardening 
	set as 0.03;
# Yield Shear
	set Vy [expr 0.55 * $fy * $Hcol1 * $twcol1];
# Shear Modulus
	set G [expr $Es/(2.0 * (1.0 + 0.30))]
# Elastic Stiffness
	set Ke [expr 0.95 * $G * $twcol1 * $Hcol1];
# Plastic Stiffness
	set Kp [expr 0.95 * $G * $Bcol1 * ($tfcol1 * $tfcol1) / $Hbeam2];

# Define Trilinear Equivalent Rotational Spring
# Yield point for Trilinear Spring at gamma1_y
	set gamma1_y [expr $Vy/$Ke]; set M1y [expr $gamma1_y * ($Ke * $Hbeam2)];
# Second Point for Trilinear Spring at 4 * gamma1_y
	set gamma2_y [expr 4.0 * $gamma1_y]; set M2y [expr $M1y + ($Kp * $Hbeam2) * ($gamma2_y - $gamma1_y)];
# Third Point for Trilinear Spring at 100 * gamma1_y
	set gamma3_y [expr 100.0 * $gamma1_y]; set M3y [expr $M2y + ($as * $Ke * $Hbeam2) * ($gamma3_y - $gamma2_y)];
    
	uniaxialMaterial Hysteretic $PanelZoneM2 $M1y $gamma1_y  $M2y $gamma2_y $M3y $gamma3_y [expr -$M1y] [expr -$gamma1_y] [expr -$M2y] [expr -$gamma2_y] [expr -$M3y] [expr -$gamma3_y] 1 1 0.0 0.0 0.0

# Trilinear Spring
# Strai Hardening 
	set as 0.03;
# Yield Shear
	set Vy [expr 0.55 * $fy * $Hcol2 * $twcol2];
# Shear Modulus
	set G [expr $Es/(2.0 * (1.0 + 0.30))]
# Elastic Stiffness
	set Ke [expr 0.95 * $G * $twcol2 * $Hcol2];
# Plastic Stiffness
	set Kp [expr 0.95 * $G * $Bcol2 * ($tfcol2 * $tfcol2) / $Hbeam2];

# Define Trilinear Equivalent Rotational Spring
# Yield point for Trilinear Spring at gamma1_y
	set gamma1_y [expr $Vy/$Ke]; set M1y [expr $gamma1_y * ($Ke * $Hbeam2)];
# Second Point for Trilinear Spring at 4 * gamma1_y
	set gamma2_y [expr 4.0 * $gamma1_y]; set M2y [expr $M1y + ($Kp * $Hbeam2) * ($gamma2_y - $gamma1_y)];
# Third Point for Trilinear Spring at 100 * gamma1_y
	set gamma3_y [expr 100.0 * $gamma1_y]; set M3y [expr $M2y + ($as * $Ke * $Hbeam2) * ($gamma3_y - $gamma2_y)];
    
	uniaxialMaterial Hysteretic $PanelZoneM3 $M1y $gamma1_y  $M2y $gamma2_y $M3y $gamma3_y [expr -$M1y] [expr -$gamma1_y] [expr -$M2y] [expr -$gamma2_y] [expr -$M3y] [expr -$gamma3_y] 1 1 0.0 0.0 0.0
   
# Trilinear Spring
# Strai Hardening 
	set as 0.03;
# Yield Shear
	set Vy [expr 0.55 * $fy * $Hcol3 * $twcol3];
# Shear Modulus
	set G [expr $Es/(2.0 * (1.0 + 0.30))]
# Elastic Stiffness
	set Ke [expr 0.95 * $G * $twcol3 * $Hcol3];
# Plastic Stiffness
	set Kp [expr 0.95 * $G * $Bcol3 * ($tfcol3 * $tfcol3) / $Hbeam1];

# Define Trilinear Equivalent Rotational Spring
# Yield point for Trilinear Spring at gamma1_y
	set gamma1_y [expr $Vy/$Ke]; set M1y [expr $gamma1_y * ($Ke * $Hbeam1)];
# Second Point for Trilinear Spring at 4 * gamma1_y
	set gamma2_y [expr 4.0 * $gamma1_y]; set M2y [expr $M1y + ($Kp * $Hbeam1) * ($gamma2_y - $gamma1_y)];
# Third Point for Trilinear Spring at 100 * gamma1_y
	set gamma3_y [expr 100.0 * $gamma1_y]; set M3y [expr $M2y + ($as * $Ke * $Hbeam1) * ($gamma3_y - $gamma2_y)];
    
	uniaxialMaterial Hysteretic $PanelZoneM4 $M1y $gamma1_y  $M2y $gamma2_y $M3y $gamma3_y [expr -$M1y] [expr -$gamma1_y] [expr -$M2y] [expr -$gamma2_y] [expr -$M3y] [expr -$gamma3_y] 1 1 0.0 0.0 0.0
  
# Trilinear Spring
# Strai Hardening 
	set as 0.03;
# Yield Shear
	set Vy [expr 0.55 * $fy * $Hcol3 * $twcol3];
# Shear Modulus
	set G [expr $Es/(2.0 * (1.0 + 0.30))]
# Elastic Stiffness
	set Ke [expr 0.95 * $G * $twcol3 * $Hcol3];
# Plastic Stiffness
	set Kp [expr 0.95 * $G * $Bcol3 * ($tfcol3 * $tfcol3) / $Hbeam2];

# Define Trilinear Equivalent Rotational Spring
# Yield point for Trilinear Spring at gamma1_y
	set gamma1_y [expr $Vy/$Ke]; set M1y [expr $gamma1_y * ($Ke * $Hbeam2)];
# Second Point for Trilinear Spring at 4 * gamma1_y
	set gamma2_y [expr 4.0 * $gamma1_y]; set M2y [expr $M1y + ($Kp * $Hbeam2) * ($gamma2_y - $gamma1_y)];
# Third Point for Trilinear Spring at 100 * gamma1_y
	set gamma3_y [expr 100.0 * $gamma1_y]; set M3y [expr $M2y + ($as * $Ke * $Hbeam2) * ($gamma3_y - $gamma2_y)];
    
	uniaxialMaterial Hysteretic $PanelZoneM5 $M1y $gamma1_y  $M2y $gamma2_y $M3y $gamma3_y [expr -$M1y] [expr -$gamma1_y] [expr -$M2y] [expr -$gamma2_y] [expr -$M3y] [expr -$gamma3_y] 1 1 0.0 0.0 0.0
    
# Trilinear Spring
# Strai Hardening 
	set as 0.03;
# Yield Shear
	set Vy [expr 0.55 * $fy * $Hcol4 * $twcol4];
# Shear Modulus
	set G [expr $Es/(2.0 * (1.0 + 0.30))]
# Elastic Stiffness
	set Ke [expr 0.95 * $G * $twcol4 * $Hcol4];
# Plastic Stiffness
	set Kp [expr 0.95 * $G * $Bcol4 * ($tfcol4* $tfcol4) / $Hbeam2];

# Define Trilinear Equivalent Rotational Spring
# Yield point for Trilinear Spring at gamma1_y
	set gamma1_y [expr $Vy/$Ke]; set M1y [expr $gamma1_y * ($Ke * $Hbeam2)];
# Second Point for Trilinear Spring at 4 * gamma1_y
	set gamma2_y [expr 4.0 * $gamma1_y]; set M2y [expr $M1y + ($Kp * $Hbeam2) * ($gamma2_y - $gamma1_y)];
# Third Point for Trilinear Spring at 100 * gamma1_y
	set gamma3_y [expr 100.0 * $gamma1_y]; set M3y [expr $M2y + ($as * $Ke * $Hbeam2) * ($gamma3_y - $gamma2_y)];
    
	uniaxialMaterial Hysteretic $PanelZoneM6 $M1y $gamma1_y  $M2y $gamma2_y $M3y $gamma3_y [expr -$M1y] [expr -$gamma1_y] [expr -$M2y] [expr -$gamma2_y] [expr -$M3y] [expr -$gamma3_y] 1 1 0.0 0.0 0.0
    
    
# puts "springs for panel zones defined"

	
###################################################################################################
#          						Define Rotational Springs for Plastic Hinges												  
###################################################################################################



# define column hinge material of moment frame
	set McMy 1.09;			# ratio of capping moment to yield moment, Mc / My
	set LS 1.505;			# basic strength deterioration (a very large # = no cyclic deterioration)
	set LK 1.505;			# unloading stiffness deterioration (a very large # = no cyclic deterioration)
	set LA 1.505;			# accelerated reloading stiffness deterioration (a very large # = no cyclic deterioration)
	set LD 1.505;			# post-capping strength deterioration 
	set cS 1.0;				# exponent for basic strength deterioration 
	set cK 1.0;				# exponent for unloading stiffness deterioration 
	set cA 1.0;				# exponent for accelerated reloading stiffness deterioration 
	set cD 1.0;				# exponent for post-capping strength deterioration
	set th_pP 0.0231;		# plastic rot capacity for pos loading
	set th_pN 0.0231;		# plastic rot capacity for neg loading
	set th_pcP 0.199;			# post-capping rot capacity for pos loading
	set th_pcN 0.199;			# post-capping rot capacity for neg loading
	set ResP 0.4;			# residual strength ratio for pos loading
	set ResN 0.4;			# residual strength ratio for neg loading
	set th_uP 0.2;			# ultimate rot capacity for pos loading
	set th_uN 0.2;			# ultimate rot capacity for neg loading
	set DP 1.0;				# rate of cyclic deterioration for pos loading
	set DN 1.0;				# rate of cyclic deterioration for neg loading
	set MyPosCol1 [expr 0.90*16324];
	set a_mem [expr ($stiffFactor1+1.0)*($MyPosCol1*($McMy-1.0)) / ($Ks_col_1*$th_pP)];	# strain hardening ratio of spring
	set b [expr ($a_mem)/(1.0+$stiffFactor1*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: Eqn B.5 is incorrect)
    
	uniaxialMaterial Bilin  $ColHingeMatM1  $Ks_col_1  $b $b $MyPosCol1 [expr -$MyPosCol1] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

	set LS 1.121;			# basic strength deterioration (a very large # = no cyclic deterioration)
	set LK 1.121;			# unloading stiffness deterioration (a very large # = no cyclic deterioration)
	set LA 1.121;			# accelerated reloading stiffness deterioration (a very large # = no cyclic deterioration)
	set LD 1.121;			# post-capping strength deterioration 
	set th_pP 0.0217;		# plastic rot capacity for pos loading
	set th_pN 0.0217;		# plastic rot capacity for neg loading
	set th_pcP 0.134;			# post-capping rot capacity for pos loading
	set th_pcN 0.134;			# post-capping rot capacity for neg loading
	set th_uP 0.2;			# ultimate rot capacity for pos loading
	set th_uN 0.2;			# ultimate rot capacity for neg loading
	set MyPosCol2 [expr 0.85*19064];
	set a_mem [expr ($stiffFactor1+1.0)*($MyPosCol2*($McMy-1.0)) / ($Ks_col_2*$th_pP)];	# strain hardening ratio of spring
	set b [expr ($a_mem)/(1.0+$stiffFactor1*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: Eqn B.5 is incorrect)
    
	uniaxialMaterial Bilin  $ColHingeMatM2  $Ks_col_2  $b $b $MyPosCol2 [expr -$MyPosCol2] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
 	
	set LS 1.619;			# basic strength deterioration (a very large # = no cyclic deterioration)
	set LK 1.619;			# unloading stiffness deterioration (a very large # = no cyclic deterioration)
	set LA 1.619;			# accelerated reloading stiffness deterioration (a very large # = no cyclic deterioration)
	set LD 1.619;			# post-capping strength deterioration 
	set th_pP 0.0235;		# plastic rot capacity for pos loading
	set th_pN 0.0235;		# plastic rot capacity for neg loading
	set th_pcP 0.178;			# post-capping rot capacity for pos loading
	set th_pcN 0.178;			# post-capping rot capacity for neg loading
	set th_uP 0.2;			# ultimate rot capacity for pos loading
	set th_uN 0.2;			# ultimate rot capacity for neg loading
	set MyPosCol3 [expr 0.95*24367.4];
	set a_mem [expr ($stiffFactor1+1.0)*($MyPosCol3*($McMy-1.0)) / ($Ks_col_3*$th_pP)];	# strain hardening ratio of spring
	set b [expr ($a_mem)/(1.0+$stiffFactor1*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: Eqn B.5 is incorrect)
    
	uniaxialMaterial Bilin  $ColHingeMatM3  $Ks_col_3 $b $b $MyPosCol3 [expr -$MyPosCol3] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
 	
	set LS 2.217;			# basic strength deterioration (a very large # = no cyclic deterioration)
	set LK 2.217;			# unloading stiffness deterioration (a very large # = no cyclic deterioration)
	set LA 2.217;			# accelerated reloading stiffness deterioration (a very large # = no cyclic deterioration)
	set LD 2.217;			# post-capping strength deterioration 
	set th_pP 0.0249;		# plastic rot capacity for pos loading
	set th_pN 0.0249;		# plastic rot capacity for neg loading
	set th_pcP 0.227;			# post-capping rot capacity for pos loading
	set th_pcN 0.227;			# post-capping rot capacity for neg loading
	set th_uP 0.26;			# ultimate rot capacity for pos loading
	set th_uN 0.26;			# ultimate rot capacity for neg loading
	set MyPosCol4 [expr 0.90*29791.3];
	set a_mem [expr ($stiffFactor1+1.0)*($MyPosCol4*($McMy-1.0)) / ($Ks_col_4*$th_pP)];	# strain hardening ratio of spring
	set b [expr ($a_mem)/(1.0+$stiffFactor1*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: Eqn B.5 is incorrect)
    
	uniaxialMaterial Bilin  $ColHingeMatM4  $Ks_col_4  $b $b $MyPosCol4 [expr -$MyPosCol4] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
 		
# puts "Column Hinges Defined"		


# Define the Beam Plastic Hinge Materials

# define beam hinge elements of moment frames

	# redefine the rotations since they are not the same
	set McMy 1.1;
	set MyPosBeam1 4102;
	set th_pP 0.0368;		# plastic rot capacity for pos loading
	set th_pN 0.0368;		# plastic rot capacity for neg loading
	set th_pcP 0.149;			# post-capping rot capacity for pos loading
	set th_pcN 0.149;			# post-capping rot capacity for neg loading
	set LS 0.7996;			# basic strength deterioration (a very large # = no cyclic deterioration)
	set LK 0.7996;			# unloading stiffness deterioration (a very large # = no cyclic deterioration)
	set LA 0.7996;			# accelerated reloading stiffness deterioration (a very large # = no cyclic deterioration)
	set LD 0.7996;			# post-capping strength deterioration 
	set a_mem [expr ($stiffFactor1+1.0)*($MyPosBeam1*($McMy-1.0)) / ($Ks_beam_1*$th_pP)];	# strain hardening ratio of spring
	set b [expr ($a_mem)/(1.0+$stiffFactor1*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: Eqn B.5 is incorrect)
	uniaxialMaterial Bilin  $BeamHingeMatM1  $Ks_beam_1 $b $b $MyPosBeam1 [expr -$MyPosBeam1] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;

	
	set MyPosBeam2 5493;
	set th_pP 0.0319;		# plastic rot capacity for pos loading
	set th_pN 0.0319;		# plastic rot capacity for neg loading
	set th_pcP 0.144;			# post-capping rot capacity for pos loading
	set th_pcN 0.144;			# post-capping rot capacity for neg loading
	set LS 0.7568;			# basic strength deterioration (a very large # = no cyclic deterioration)
	set LK 0.7568;			# unloading stiffness deterioration (a very large # = no cyclic deterioration)
	set LA 0.7568;			# accelerated reloading stiffness deterioration (a very large # = no cyclic deterioration)
	set LD 0.7568;			# post-capping strength deterioration 
	set a_mem [expr ($stiffFactor1+1.0)*($MyPosBeam2*($McMy-1.0)) / ($Ks_beam_2*$th_pP)];	# strain hardening ratio of spring
	set b [expr ($a_mem)/(1.0+$stiffFactor1*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: Eqn B.5 is incorrect)
	uniaxialMaterial Bilin  $BeamHingeMatM2  $Ks_beam_2 $b $b $MyPosBeam2 [expr -$MyPosBeam2] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
     
	
	set MyPosBeam3 7025;
	set th_pP 0.0336;		# plastic rot capacity for pos loading
	set th_pN 0.0336;		# plastic rot capacity for neg loading
	set th_pcP 0.180;			# post-capping rot capacity for pos loading
	set th_pcN 0.180;			# post-capping rot capacity for neg loading
	set LS 0.9852;			# basic strength deterioration (a very large # = no cyclic deterioration)
	set LK 0.9852;			# unloading stiffness deterioration (a very large # = no cyclic deterioration)
	set LA 0.9852;			# accelerated reloading stiffness deterioration (a very large # = no cyclic deterioration)
	set LD 0.9850;			# post-capping strength deterioration 
	set a_mem [expr ($stiffFactor1+1.0)*($MyPosBeam3*($McMy-1.0)) / ($Ks_beam_3*$th_pP)];	# strain hardening ratio of spring
	set b [expr ($a_mem)/(1.0+$stiffFactor1*(1.0-$a_mem))];							# modified strain hardening ratio of spring (Ibarra & Krawinkler 2005, note: Eqn B.5 is incorrect)
	uniaxialMaterial Bilin  $BeamHingeMatM3  $Ks_beam_3 $b $b $MyPosBeam3 [expr -$MyPosBeam3] $LS $LK $LA $LD $cS $cK $cA $cD $th_pP $th_pN $th_pcP $th_pcN $ResP $ResN $th_uP $th_uN $DP $DN;
     
# puts "Beam Hinges Defined"			

	uniaxialMaterial Elastic $rigidMat 1e10;
	uniaxialMaterial Elastic $pinnedMat 1e-10;
#########################################################################################################################
### DEFINE NODES
 
	

	 node   201011  $Pier1 $Floor1 
     node   201013  $Pier1 $Floor1 
     node   201021  $Pier2 $Floor1 
     node   201023  $Pier2 $Floor1 
     node   201031  $Pier3 $Floor1 
     node   201033  $Pier3 $Floor1 
     node   201041  $Pier4 $Floor1 
     node   201043  $Pier4 $Floor1 	
	 
# Define the nodes around all of the joints in the primary frame
      #   nodeNum   X   Y

     node   206021  $Pier2 [expr $Floor6-$phvert1] 
     node   206022  [expr $Pier2 + $pzlat1] $Floor6 
     node   206023  $Pier2 [expr $Floor6+$phvert1]
     node   206024  [expr $Pier2 - $pzlat1] $Floor6 
     node   206031  $Pier3 [expr $Floor6-$phvert1] 
     node   206032  [expr $Pier3 + $pzlat1] $Floor6 
     node   206033  $Pier3 [expr $Floor6+$phvert1]
     node   206034  [expr $Pier3 - $pzlat1] $Floor6 
     node   206041  $Pier4 [expr $Floor6-$phvert1] 
     node   206042  [expr $Pier4 + $pzlat1] $Floor6 
     node   206043  $Pier4 [expr $Floor6+$phvert1] 
     node   206044  [expr $Pier4 - $pzlat1] $Floor6

     node   205021  $Pier2 [expr $Floor5-$phvert2] 
     node   205022  [expr $Pier2 + $pzlat2] $Floor5 
     node   205023  $Pier2 [expr $Floor5+$phvert2]
     node   205024  [expr $Pier2 - $pzlat2] $Floor5 
     node   205031  $Pier3 [expr $Floor5-$phvert2] 
     node   205032  [expr $Pier3 + $pzlat2] $Floor5 
     node   205033  $Pier3 [expr $Floor5+$phvert2]
     node   205034  [expr $Pier3 - $pzlat2] $Floor5 
     node   205041  $Pier4 [expr $Floor5-$phvert2] 
     node   205042  [expr $Pier4 + $pzlat2] $Floor5 
     node   205043  $Pier4 [expr $Floor5+$phvert2] 
     node   205044  [expr $Pier4 - $pzlat2] $Floor5
 
     node   204011  $Pier1 [expr $Floor4-$phvert2]
     node   204012  [expr $Pier1 + $pzlat2] $Floor4 
     node   204013  $Pier1 [expr $Floor4+$phvert2] 
     node   204014  [expr $Pier1 - $pzlat2] $Floor4 
     node   204021  $Pier2 [expr $Floor4-$phvert2] 
     node   204022  [expr $Pier2 + $pzlat2] $Floor4 
     node   204023  $Pier2 [expr $Floor4+$phvert2] 
     node   204024  [expr $Pier2 - $pzlat2] $Floor4 
     node   204031  $Pier3 [expr $Floor4-$phvert2]
     node   204032  [expr $Pier3 + $pzlat2] $Floor4 
     node   204033  $Pier3 [expr $Floor4+$phvert2] 
     node   204034  [expr $Pier3 - $pzlat2] $Floor4
     node   204041  $Pier4 [expr $Floor4-$phvert2] 
     node   204042  [expr $Pier4 + $pzlat2] $Floor4 
     node   204043  $Pier4 [expr $Floor4+$phvert2] 
     node   204044  [expr $Pier4 - $pzlat2] $Floor4 
 
     node   203011  $Pier1 [expr $Floor3-$phvert2] 
     node   203012  [expr $Pier1 + $pzlat2] $Floor3 
     node   203013  $Pier1 [expr $Floor3+$phvert2]
     node   203014  [expr $Pier1 - $pzlat2] $Floor3 
     node   203021  $Pier2 [expr $Floor3-$phvert2] 
     node   203022  [expr $Pier2 + $pzlat2] $Floor3 
     node   203023  $Pier2 [expr $Floor3+$phvert2] 
     node   203024  [expr $Pier2 - $pzlat2] $Floor3 
     node   203031  $Pier3 [expr $Floor3-$phvert2] 
     node   203032  [expr $Pier3 + $pzlat2] $Floor3 
     node   203033  $Pier3 [expr $Floor3+$phvert2] 
     node   203034  [expr $Pier3 - $pzlat2] $Floor3 
     node   203041  $Pier4 [expr $Floor3-$phvert2] 
     node   203042  [expr $Pier4 + $pzlat2] $Floor3 
     node   203043  $Pier4 [expr $Floor3+$phvert2]
     node   203044  [expr $Pier4 - $pzlat2] $Floor3 
 
     node   202011  $Pier1 [expr $Floor2-$phvert2] 
     node   202012  [expr $Pier1 + $pzlat2] $Floor2 
     node   202013  $Pier1 [expr $Floor2+$phvert2] 
     node   202014  [expr $Pier1 - $pzlat2] $Floor2 
     node   202021  $Pier2 [expr $Floor2-$phvert2]
     node   202022  [expr $Pier2 + $pzlat2] $Floor2 
     node   202023  $Pier2 [expr $Floor2+$phvert2] 
     node   202024  [expr $Pier2 - $pzlat2] $Floor2 
     node   202031  $Pier3 [expr $Floor2-$phvert2]
     node   202032  [expr $Pier3 + $pzlat2] $Floor2 
     node   202033  $Pier3 [expr $Floor2+$phvert2] 
     node   202034  [expr $Pier3 - $pzlat2] $Floor2 
     node   202041  $Pier4 [expr $Floor2-$phvert2] 
     node   202042  [expr $Pier4 + $pzlat2] $Floor2 
     node   202043  $Pier4 [expr $Floor2+$phvert2] 
     node   202044  [expr $Pier4 - $pzlat2] $Floor2 


     node   206026  [expr $Pier2 + $phlat1] $Floor6 
     node   206036  [expr $Pier3 + $phlat1] $Floor6 
     node   206038  [expr $Pier3 - $phlat1] $Floor6 
     node   206048  [expr $Pier4 - $phlat1] $Floor6		 

     node   205026  [expr $Pier2 + $phlat2] $Floor5 
     node   205036  [expr $Pier3 + $phlat2] $Floor5 
     node   205038  [expr $Pier3 - $phlat2] $Floor5 
     node   205048  [expr $Pier4 - $phlat2] $Floor5
 
     node   204016  [expr $Pier1 + $phlat2] $Floor4 
     node   204026  [expr $Pier2 + $phlat2] $Floor4 
     node   204028  [expr $Pier2 - $phlat2] $Floor4 
     node   204036  [expr $Pier3 + $phlat2] $Floor4 
     node   204038  [expr $Pier3 - $phlat2] $Floor4
     node   204048  [expr $Pier4 - $phlat2] $Floor4 
 
     node   203016  [expr $Pier1 + $phlat2] $Floor3 
     node   203026  [expr $Pier2 + $phlat2] $Floor3 
     node   203028  [expr $Pier2 - $phlat2] $Floor3 
     node   203036  [expr $Pier3 + $phlat2] $Floor3 
     node   203038  [expr $Pier3 - $phlat2] $Floor3 
     node   203048  [expr $Pier4 - $phlat2] $Floor3 
 
     node   202016  [expr $Pier1 + $phlat2] $Floor2 
     node   202026  [expr $Pier2 + $phlat2] $Floor2 
     node   202028  [expr $Pier2 - $phlat2] $Floor2 
     node   202036  [expr $Pier3 + $phlat2] $Floor2 
     node   202038  [expr $Pier3 - $phlat2] $Floor2 
     node   202048  [expr $Pier4 - $phlat2] $Floor2 
	 
     node   206027  [expr $Pier2 + $phlat1] $Floor6 
     node   206037  [expr $Pier3 + $phlat1] $Floor6 
     node   206039  [expr $Pier3 - $phlat1] $Floor6 
     node   206049  [expr $Pier4 - $phlat1] $Floor6
	 
     node   205027  [expr $Pier2 + $phlat2] $Floor5 
     node   205037  [expr $Pier3 + $phlat2] $Floor5 
     node   205039  [expr $Pier3 - $phlat2] $Floor5 
     node   205049  [expr $Pier4 - $phlat2] $Floor5
 
     node   204017  [expr $Pier1 + $phlat2] $Floor4 
     node   204027  [expr $Pier2 + $phlat2] $Floor4 
     node   204029  [expr $Pier2 - $phlat2] $Floor4 
     node   204037  [expr $Pier3 + $phlat2] $Floor4 
     node   204039  [expr $Pier3 - $phlat2] $Floor4
     node   204049  [expr $Pier4 - $phlat2] $Floor4 
 
     node   203017  [expr $Pier1 + $phlat2] $Floor3 
     node   203027  [expr $Pier2 + $phlat2] $Floor3 
     node   203029  [expr $Pier2 - $phlat2] $Floor3 
     node   203037  [expr $Pier3 + $phlat2] $Floor3 
     node   203039  [expr $Pier3 - $phlat2] $Floor3 
     node   203049  [expr $Pier4 - $phlat2] $Floor3 
 
     node   202017  [expr $Pier1 + $phlat2] $Floor2 
     node   202027  [expr $Pier2 + $phlat2] $Floor2 
     node   202029  [expr $Pier2 - $phlat2] $Floor2 
     node   202037  [expr $Pier3 + $phlat2] $Floor2 
     node   202039  [expr $Pier3 - $phlat2] $Floor2 
     node   202049  [expr $Pier4 - $phlat2] $Floor2 
	 
# Define the nodes around all of the joints in the primary frame

     node   3061  $Pier5 $Floor6
     node   3051  $Pier5 $Floor5
	 node   3053  $Pier5 $Floor5
	 node   3041  $Pier5 $Floor4
	 node   3043  $Pier5 $Floor4
     node   3031  $Pier5 $Floor3
	 node   3033  $Pier5 $Floor3
     node   3021  $Pier5 $Floor2
	 node   3023  $Pier5 $Floor2
     node   3013  $Pier5 $Floor1
	 
	 # puts "nodes defined"
#########################################################################################################################
### DEFINE FIXITIES
 
# Define the fixities at the bases of the columns of the primary lateral frame
#       node    DX DY RZ
     fix    201011    1  1  1 
     fix    201021    1  1  1 
     fix    201031    1  1  1 
     fix    201041    1  1  1 
 
# Define the fixity at the baseof the leaning column
#       node    DX DY RZ
     fix    3013    1  1  0 
 		
     # puts "fixitiy defined"		
#########################################################################################################################
### DEFINE ELEMENTS
 
# Define the column elements
             ##                 tag     node1   node2       A      Es    EI          geomTransf  

     element elasticBeamColumn   30502 205023 206021   $Acol1  $Es   $Icol_mod1  $primaryGeomTransT 
     element elasticBeamColumn   30503 205033 206031   $Acol3  $Es   $Icol_mod3  $primaryGeomTransT 
     element elasticBeamColumn   30504 205043 206041   $Acol1  $Es   $Icol_mod1  $primaryGeomTransT
	 
     element elasticBeamColumn   30402 204023 205021   $Acol1  $Es   $Icol_mod1  $primaryGeomTransT 
     element elasticBeamColumn   30403 204033 205031   $Acol3  $Es   $Icol_mod3  $primaryGeomTransT 
     element elasticBeamColumn   30404 204043 205041   $Acol1  $Es   $Icol_mod1  $primaryGeomTransT 
 
     element elasticBeamColumn   30301 203013 204011   $Acol1  $Es   $Icol_mod1  $primaryGeomTransT 
     element elasticBeamColumn   30302 203023 204021   $Acol4  $Es   $Icol_mod4  $primaryGeomTransT 
     element elasticBeamColumn   30303 203033 204031   $Acol4  $Es   $Icol_mod4  $primaryGeomTransT 
     element elasticBeamColumn   30304 203043 204041   $Acol2  $Es   $Icol_mod2  $primaryGeomTransT 
 
     element elasticBeamColumn   30201 202013 203011   $Acol1  $Es   $Icol_mod1  $primaryGeomTransT 
     element elasticBeamColumn   30202 202023 203021   $Acol4  $Es   $Icol_mod4  $primaryGeomTransT 
     element elasticBeamColumn   30203 202033 203031   $Acol4  $Es   $Icol_mod4  $primaryGeomTransT 
     element elasticBeamColumn   30204 202043 203041   $Acol2  $Es   $Icol_mod2  $primaryGeomTransT 
 
     element elasticBeamColumn   30101 201013 202011   $Acol1  $Es   $Icol_mod1  $primaryGeomTransT 
     element elasticBeamColumn   30102 201023 202021   $Acol4  $Es   $Icol_mod4  $primaryGeomTransT 
     element elasticBeamColumn   30103 201033 202031   $Acol4  $Es   $Icol_mod4  $primaryGeomTransT 
     element elasticBeamColumn   30104 201043 202041   $Acol2  $Es   $Icol_mod2  $primaryGeomTransT 
 	
    # puts "columns defined"		
# Define the beam elements
             ##                 tag     node1   node2       EA      Es    EI          geomTransf  
 
     element elasticBeamColumn   20604 206022 206026   $Abeam1  $Es   $Ibeam_mod1  $primaryGeomTransT 
     element elasticBeamColumn   20605 206027 206038   $Abeam1  $Es   $Ibeam_mod1  $primaryGeomTransT 	 
     element elasticBeamColumn   20606 206039 206034   $Abeam1  $Es   $Ibeam_mod1  $primaryGeomTransT 
     element elasticBeamColumn   20607 206032 206036   $Abeam1  $Es   $Ibeam_mod1  $primaryGeomTransT 
     element elasticBeamColumn   20608 206037 206048   $Abeam1  $Es   $Ibeam_mod1  $primaryGeomTransT 
     element elasticBeamColumn   20609 206049 206044   $Abeam1  $Es   $Ibeam_mod1  $primaryGeomTransT 
	 rotSpring2D     6063    206026  206027 $BeamHingeMatM1
	 rotSpring2D     6064    206038  206039 $BeamHingeMatM1
	 rotSpring2D     6065    206036  206037 $BeamHingeMatM1
	 rotSpring2D     6066    206048  206049 $BeamHingeMatM1


     element elasticBeamColumn   20504 205022 205026   $Abeam2  $Es   $Ibeam_mod2  $primaryGeomTransT 
     element elasticBeamColumn   20505 205027 205038   $Abeam2  $Es   $Ibeam_mod2  $primaryGeomTransT 
     element elasticBeamColumn   20506 205039 205034   $Abeam2  $Es   $Ibeam_mod2  $primaryGeomTransT 
     element elasticBeamColumn   20507 205032 205036   $Abeam2  $Es   $Ibeam_mod2  $primaryGeomTransT 
     element elasticBeamColumn   20508 205037 205048   $Abeam2  $Es   $Ibeam_mod2  $primaryGeomTransT 
     element elasticBeamColumn   20509 205049 205044   $Abeam2  $Es   $Ibeam_mod2  $primaryGeomTransT 
	 rotSpring2D     6053    205026  205027 $BeamHingeMatM2
	 rotSpring2D     6054    205038  205039 $BeamHingeMatM2
	 rotSpring2D     6055    205036  205037 $BeamHingeMatM2
	 rotSpring2D     6056    205048  205049 $BeamHingeMatM2
	 
	
     element elasticBeamColumn   20401 204012 204016   $Abeam2  $Es   $Ibeam_mod2  $primaryGeomTransT 
     element elasticBeamColumn   20402 204017 204028   $Abeam2  $Es   $Ibeam_mod2  $primaryGeomTransT 
     element elasticBeamColumn   20403 204029 204024   $Abeam2  $Es   $Ibeam_mod2  $primaryGeomTransT 
     element elasticBeamColumn   20404 204022 204026   $Abeam2  $Es   $Ibeam_mod2  $primaryGeomTransT 
     element elasticBeamColumn   20405 204027 204038   $Abeam2  $Es   $Ibeam_mod2  $primaryGeomTransT 
     element elasticBeamColumn   20406 204039 204034   $Abeam2  $Es   $Ibeam_mod2  $primaryGeomTransT 
     element elasticBeamColumn   20407 204032 204036   $Abeam2  $Es   $Ibeam_mod2  $primaryGeomTransT 
     element elasticBeamColumn   20408 204037 204048   $Abeam2  $Es   $Ibeam_mod2  $primaryGeomTransT 
     element elasticBeamColumn   20409 204049 204044   $Abeam2  $Es   $Ibeam_mod2  $primaryGeomTransT 
 	 rotSpring2D     6041    204016  204017 $BeamHingeMatM2
	 rotSpring2D     6042    204028  204029 $BeamHingeMatM2
	 rotSpring2D     6043    204026  204027 $BeamHingeMatM2
	 rotSpring2D     6044    204038  204039 $BeamHingeMatM2
	 rotSpring2D     6045    204036  204037 $BeamHingeMatM2
	 rotSpring2D     6046    204048  204049 $BeamHingeMatM2


	 
     element elasticBeamColumn   20301 203012 203016   $Abeam3  $Es   $Ibeam_mod3  $primaryGeomTransT 
     element elasticBeamColumn   20302 203017 203028   $Abeam3  $Es   $Ibeam_mod3  $primaryGeomTransT 
     element elasticBeamColumn   20303 203029 203024   $Abeam3  $Es   $Ibeam_mod3  $primaryGeomTransT 
     element elasticBeamColumn   20304 203022 203026   $Abeam3  $Es   $Ibeam_mod3  $primaryGeomTransT 
     element elasticBeamColumn   20305 203027 203038   $Abeam3  $Es   $Ibeam_mod3  $primaryGeomTransT 
     element elasticBeamColumn   20306 203039 203034   $Abeam3  $Es   $Ibeam_mod3  $primaryGeomTransT 
     element elasticBeamColumn   20307 203032 203036   $Abeam3  $Es   $Ibeam_mod3  $primaryGeomTransT 
     element elasticBeamColumn   20308 203037 203048   $Abeam3  $Es   $Ibeam_mod3  $primaryGeomTransT 
     element elasticBeamColumn   20309 203049 203044   $Abeam3  $Es   $Ibeam_mod3  $primaryGeomTransT 
	 rotSpring2D     6031    203016  203017 $BeamHingeMatM3
	 rotSpring2D     6032    203028  203029 $BeamHingeMatM3
	 rotSpring2D     6033    203026  203027 $BeamHingeMatM3
	 rotSpring2D     6034    203038  203039 $BeamHingeMatM3
	 rotSpring2D     6035    203036  203037 $BeamHingeMatM3
	 rotSpring2D     6036    203048  203049 $BeamHingeMatM3


	 
     element elasticBeamColumn   20201 202012 202016   $Abeam3  $Es   $Ibeam_mod3  $primaryGeomTransT 
     element elasticBeamColumn   20202 202017 202028   $Abeam3  $Es   $Ibeam_mod3  $primaryGeomTransT 
     element elasticBeamColumn   20203 202029 202024   $Abeam3  $Es   $Ibeam_mod3  $primaryGeomTransT 
     element elasticBeamColumn   20204 202022 202026   $Abeam3  $Es   $Ibeam_mod3  $primaryGeomTransT 
     element elasticBeamColumn   20205 202027 202038   $Abeam3  $Es   $Ibeam_mod3  $primaryGeomTransT 
     element elasticBeamColumn   20206 202039 202034   $Abeam3  $Es   $Ibeam_mod3  $primaryGeomTransT 
     element elasticBeamColumn   20207 202032 202036   $Abeam3  $Es   $Ibeam_mod3  $primaryGeomTransT 
     element elasticBeamColumn   20208 202037 202048   $Abeam3  $Es   $Ibeam_mod3  $primaryGeomTransT 
     element elasticBeamColumn   20209 202049 202044   $Abeam3  $Es   $Ibeam_mod3  $primaryGeomTransT 
	 rotSpring2D     6021    202016  202017 $BeamHingeMatM3
	 rotSpring2D     6022    202028  202029 $BeamHingeMatM3
	 rotSpring2D     6023    202026  202027 $BeamHingeMatM3
	 rotSpring2D     6024    202038  202039 $BeamHingeMatM3
	 rotSpring2D     6025    202036  202037 $BeamHingeMatM3
	 rotSpring2D     6026    202048  202049 $BeamHingeMatM3
	 
		 
	 
 	# Define the column base rotational spring (both elastic foundation springs and column base PH springs)
     ##  eleID nodeR nodeC matID 
     rotSpring2D     6012    201011  201013  $ColHingeMatM1 
     rotSpring2D     6014    201021  201023  $ColHingeMatM4 
     rotSpring2D     6016    201031  201033  $ColHingeMatM4 
     rotSpring2D     6018    201041  201043  $ColHingeMatM2 		
	 
 	# puts "beams defined"
# Define the joint elements
    #                  tag         n1       n2     n3     n4       centerNode  PH1     PH2     PH3     PH4     shearPanel  largeDisp
 

     element Joint2D     40602       206021  206022  206023  206024  206025      $ColHingeMatM1    0    0   0  $PanelZoneM1       $lrgDsp 
     element Joint2D     40603       206031  206032  206033  206034  206035      $ColHingeMatM3    0    0   0  $PanelZoneM4       $lrgDsp 
     element Joint2D     40604       206041  206042  206043  206044  206045      $ColHingeMatM1    0    0   0  $PanelZoneM1       $lrgDsp 	

     element Joint2D     40502       205021  205022  205023  205024  205025      $ColHingeMatM1    0    $ColHingeMatM1   0  $PanelZoneM2       $lrgDsp 
     element Joint2D     40503       205031  205032  205033  205034  205035      $ColHingeMatM3    0    $ColHingeMatM3   0  $PanelZoneM5       $lrgDsp 
     element Joint2D     40504       205041  205042  205043  205044  205045      $ColHingeMatM1    0    $ColHingeMatM1   0  $PanelZoneM2       $lrgDsp 
 
     element Joint2D     40401       204011  204012  204013  204014  204015      $ColHingeMatM1    0   0   0  $PanelZoneM2       $lrgDsp 
     element Joint2D     40402       204021  204022  204023  204024  204025      $ColHingeMatM4    0   $ColHingeMatM1   0  $PanelZoneM6       $lrgDsp 
     element Joint2D     40403       204031  204032  204033  204034  204035      $ColHingeMatM4    0   $ColHingeMatM3   0  $PanelZoneM6       $lrgDsp 
     element Joint2D     40404       204041  204042  204043  204044  204045      $ColHingeMatM2    0   $ColHingeMatM1   0  $PanelZoneM3       $lrgDsp 
 
     element Joint2D     40301       203011  203012  203013  203014  203015      $ColHingeMatM1    0   $ColHingeMatM1   0  $PanelZoneM2       $lrgDsp 
     element Joint2D     40302       203021  203022  203023  203024  203025      $ColHingeMatM4    0   $ColHingeMatM4   0  $PanelZoneM6       $lrgDsp 
     element Joint2D     40303       203031  203032  203033  203034  203035      $ColHingeMatM4    0   $ColHingeMatM4   0  $PanelZoneM6       $lrgDsp 
     element Joint2D     40304       203041  203042  203043  203044  203045      $ColHingeMatM2    0   $ColHingeMatM2   0  $PanelZoneM3       $lrgDsp 
 
     element Joint2D     40201       202011  202012  202013  202014  202015      $ColHingeMatM1    0   $ColHingeMatM1   0  $PanelZoneM2       $lrgDsp 
     element Joint2D     40202       202021  202022  202023  202024  202025      $ColHingeMatM4    0   $ColHingeMatM4   0  $PanelZoneM6       $lrgDsp 
     element Joint2D     40203       202031  202032  202033  202034  202035      $ColHingeMatM4    0   $ColHingeMatM4   0  $PanelZoneM6       $lrgDsp 
     element Joint2D     40204       202041  202042  202043  202044  202045      $ColHingeMatM2    0   $ColHingeMatM2   0  $PanelZoneM3       $lrgDsp 
 	
	# puts "joints defined"	
# Define the leaning column and horizintal rigid link elements
     element elasticBeamColumn   5051    3053  3061  $A_strut    $Es    $I_strut     $primaryGeomTransT 
     element elasticBeamColumn   5041    3043  3051  $A_strut    $Es    $I_strut     $primaryGeomTransT 
     element elasticBeamColumn   5031    3033  3041  $A_strut    $Es    $I_strut     $primaryGeomTransT 
     element elasticBeamColumn   5021    3023  3031  $A_strut    $Es    $I_strut     $primaryGeomTransT 
     element elasticBeamColumn   5011    3013  3021  $A_strut    $Es    $I_strut     $primaryGeomTransT 
	 
	 equalDOF 3051 3053 1 2;
	 equalDOF 3041 3043 1 2;
	 equalDOF 3031 3033 1 2;
	 equalDOF 3021 3023 1 2;
	 
	 # equalDOF 206042 3061 1;
	 # equalDOF 205042 3051 1;
	 # equalDOF 204042 3041 1;
	 # equalDOF 203042 3031 1;
	 # equalDOF 202042 3021 1;
	 
     element truss           5062    3061 206042        $A_strut    $strutMatT
     element truss           5052    3051 205042        $A_strut    $strutMatT
     element truss           5042    3041 204042        $A_strut    $strutMatT
     element truss           5032    3031 203042        $A_strut    $strutMatT
     element truss           5022    3021 202042        $A_strut    $strutMatT
 	
	# puts "leaning columns defined"	
	
	

##############################################################################################################################
#          										   Define Building Masses										             #
##############################################################################################################################
 
# calculate nodal masses -- lump floor masses at moment frame nodes	

	set MFloorWeight2 137.14;										# Weight of typical floor in kips
	set MFloorWeight3 185.32;										# Weight of typical floor in kips
	set MFloorWeight4 197.16;										# Weight of typical floor in kips
	set MFloorWeight5 140.89;										# Weight of typical floor in kips
	set MFloorWeight6 141.42;										# Weight of typical floor in kips
	set MFloorMass2 [expr $MFloorWeight2/$g];											# Mass of typical floor in kips*s^2/in
	set MFloorMass3 [expr $MFloorWeight3/$g];											# Mass of typical floor in kips*s^2/in
	set MFloorMass4 [expr $MFloorWeight4/$g];											# Mass of typical floor in kips*s^2/in
	set MFloorMass5 [expr $MFloorWeight5/$g];											# Mass of typical floor in kips*s^2/in
	set MFloorMass6 [expr $MFloorWeight6/$g];											# Mass of typical floor in kips*s^2/in
	
# calculate nodal masses -- lump floor masses at gravity frame nodes	

	set GFloorWeight2 396.46;										# Weight of typical floor in kips
	set GFloorWeight3 359.98;										# Weight of typical floor in kips
	set GFloorWeight4 404.59;										# Weight of typical floor in kips
	set GFloorWeight5 275.71;										# Weight of typical floor in kips
	set GFloorWeight6 311.71;										# Weight of typical floor in kips
	set GFloorMass2 [expr $GFloorWeight2/$g];											# Mass of typical floor in kips*s^2/in
	set GFloorMass3 [expr $GFloorWeight3/$g];											# Mass of typical floor in kips*s^2/in
	set GFloorMass4 [expr $GFloorWeight4/$g];											# Mass of typical floor in kips*s^2/in
	set GFloorMass5 [expr $GFloorWeight5/$g];											# Mass of typical floor in kips*s^2/in
	set GFloorMass6 [expr $GFloorWeight6/$g];											# Mass of typical floor in kips*s^2/in
	
	# puts "nodal masses defined"


#########################################################################################################################
### DEFINE GRAVITY LOADS
 
     pattern Plain 1 Linear {
             # Distributed beam element loads
             # eleLoad -ele   20604 20515 20525 20606 20607 20608 20609 -type -beamUniform   [expr -$MFloorWeight6/2/$WBay1]
             # eleLoad -ele   20504 20515 20525 20506 20507 20508 20509 -type -beamUniform   [expr -$MFloorWeight5/2/$WBay1]	
			 # eleLoad -ele   20401 20402 20403 20404 20415 20425 20406 20407 20408 20409 -type -beamUniform   [expr -$MFloorWeight4/3/$WBay1]		
			 # eleLoad -ele   20301 20302 20303 20304 20315 20325 20306 20307 20308 20309 -type -beamUniform   [expr -$MFloorWeight3/3/$WBay1]	 
			 # eleLoad -ele   20201 20202 20203 20204 20215 20225 20206 20207 20208 20209 -type -beamUniform   [expr -$MFloorWeight2/3/$WBay1]

             # Point gravity loads on moment column
             #        node    X   Y           M
             load    206023    0.0    [expr -29.31]   0.0			 
             load    206033    0.0    [expr -57.49]   0.0
             load    206043    0.0    [expr -54.62]   0.0
             load    205023    0.0    [expr -32.07]   0.0			 
             load    205033    0.0    [expr -55.34]   0.0
             load    205043    0.0    [expr -53.48]   0.0		
             load    204013    0.0    [expr -30.52]   0.0			 
             load    204023    0.0    [expr -58.99]   0.0			 
             load    204033    0.0    [expr -54.24]   0.0
             load    204043    0.0    [expr -53.41]   0.0	
             load    203013    0.0    [expr -26.57]   0.0			 
             load    203023    0.0    [expr -50.57]   0.0			 
             load    203033    0.0    [expr -54.71]   0.0
             load    203043    0.0    [expr -53.47]   0.0	
             load    202013    0.0    [expr -26.35]   0.0			 
             load    202023    0.0    [expr -42.93]   0.0			 
             load    202033    0.0    [expr -38.38]   0.0
             load    202043    0.0    [expr -29.48]   0.0	
			 
             # Point gravity loads on leaning column
             #        node    X   Y           M
             load    3061    0.0    [expr -$GFloorWeight6]   0.0			 
             load    3051    0.0    [expr -$GFloorWeight5]   0.0
             load    3041    0.0    [expr -$GFloorWeight4]   0.0
             load    3031    0.0    [expr -$GFloorWeight3]   0.0
             load    3021    0.0    [expr -$GFloorWeight2]   0.0	
      }
 
 
#########################################################################################################################
### DEFINE MASSES

set mass_mom_joint 65.0; # Inertial mass per joint
 
# Define small masses at all nodes as the column-foundation connection
      #       nodeNum    X    Y     Rotation
     mass   201013    0.0   0.0   [expr {$mass_mom_joint/30}]  
     mass   201023    0.0   0.0   [expr {$mass_mom_joint/30}]  
     mass   201033    0.0   0.0   [expr {$mass_mom_joint/30}]  
     mass   201043    0.0   0.0   [expr {$mass_mom_joint/30}]  
 
# Define small masses as the nodes around all of the joints in the primary frame
	 mass   3061    $GFloorMass6   $GFloorMass6   [expr {$mass_mom_joint*2.5}]  
     mass   3051    [expr {$GFloorMass5/2}]   [expr {$GFloorMass5/2}]   [expr {$mass_mom_joint*2.5/2}]  
     mass   3053    [expr {$GFloorMass5/2}]   [expr {$GFloorMass5/2}]   [expr {$mass_mom_joint*2.5/2}]  
	 mass   3041    [expr {$GFloorMass4/2}]   [expr {$GFloorMass4/2}]   [expr {$mass_mom_joint*2.5/2}]  
	 mass   3043    [expr {$GFloorMass4/2}]   [expr {$GFloorMass4/2}]   [expr {$mass_mom_joint*2.5/2}]  
	 mass   3031    [expr {$GFloorMass3/2}]   [expr {$GFloorMass3/2}]   [expr {$mass_mom_joint*2.5/2}]  
	 mass   3033    [expr {$GFloorMass3/2}]   [expr {$GFloorMass3/2}]   [expr {$mass_mom_joint*2.5/2}]  
	 mass   3021    [expr {$GFloorMass2/2}]   [expr {$GFloorMass2/2}]   [expr {$mass_mom_joint*2.5/2}]  
	 mass   3023    [expr {$GFloorMass2/2}]   [expr {$GFloorMass2/2}]   [expr {$mass_mom_joint*2.5/2}]  
	 mass   3013    0.0   0.0   [expr {$mass_mom_joint/30}]  
 
# Define building masses
      #       nodeNum    X    Y     Rotation

     mass    206021       [expr ($MFloorMass6)/($NMBays)/6]       	[expr ($MFloorMass6)/($NMBays)/6]             [expr {$mass_mom_joint/6}]
     mass    206022       [expr ($MFloorMass6)/($NMBays)/6]       	[expr ($MFloorMass6)/($NMBays)/6]             [expr {$mass_mom_joint/6}]
     mass    206023       [expr ($MFloorMass6)/($NMBays)/6]       	[expr ($MFloorMass6)/($NMBays)/6]             [expr {$mass_mom_joint/6}]
     mass    206024       [expr ($MFloorMass6)/($NMBays)/6]       	[expr ($MFloorMass6)/($NMBays)/6]             [expr {$mass_mom_joint/6}]
     mass    206031       [expr ($MFloorMass6)/($NMBays)/8]       	[expr ($MFloorMass6)/($NMBays)/8]             [expr {$mass_mom_joint/8}]
     mass    206032       [expr ($MFloorMass6)/($NMBays)/8]       	[expr ($MFloorMass6)/($NMBays)/8]             [expr {$mass_mom_joint/8}]
     mass    206033       [expr ($MFloorMass6)/($NMBays)/8]       	[expr ($MFloorMass6)/($NMBays)/8]             [expr {$mass_mom_joint/8}]
     mass    206034       [expr ($MFloorMass6)/($NMBays)/8]       	[expr ($MFloorMass6)/($NMBays)/8]             [expr {$mass_mom_joint/8}]
     mass    206041       [expr ($MFloorMass6)/($NMBays)/6]       	[expr ($MFloorMass6)/($NMBays)/6]             [expr {$mass_mom_joint/6}]
     mass    206042       [expr ($MFloorMass6)/($NMBays)/6]       	[expr ($MFloorMass6)/($NMBays)/6]             [expr {$mass_mom_joint/6}]
     mass    206043       [expr ($MFloorMass6)/($NMBays)/6]       	[expr ($MFloorMass6)/($NMBays)/6]             [expr {$mass_mom_joint/6}]
     mass    206044       [expr ($MFloorMass6)/($NMBays)/6]       	[expr ($MFloorMass6)/($NMBays)/6]             [expr {$mass_mom_joint/6}]	

     mass    205021       [expr ($MFloorMass5)/($NMBays)/6]       	[expr ($MFloorMass5)/($NMBays)/6]             [expr {$mass_mom_joint/6}]
     mass    205022       [expr ($MFloorMass5)/($NMBays)/6]       	[expr ($MFloorMass5)/($NMBays)/6]             [expr {$mass_mom_joint/6}]
     mass    205023       [expr ($MFloorMass5)/($NMBays)/6]       	[expr ($MFloorMass5)/($NMBays)/6]             [expr {$mass_mom_joint/6}]
     mass    205024       [expr ($MFloorMass5)/($NMBays)/6]       	[expr ($MFloorMass5)/($NMBays)/6]             [expr {$mass_mom_joint/6}]
     mass    205031       [expr ($MFloorMass5)/($NMBays)/8]       	[expr ($MFloorMass5)/($NMBays)/8]             [expr {$mass_mom_joint/8}]
     mass    205032       [expr ($MFloorMass5)/($NMBays)/8]       	[expr ($MFloorMass5)/($NMBays)/8]             [expr {$mass_mom_joint/8}]
     mass    205033       [expr ($MFloorMass5)/($NMBays)/8]       	[expr ($MFloorMass5)/($NMBays)/8]             [expr {$mass_mom_joint/8}]
     mass    205034       [expr ($MFloorMass5)/($NMBays)/8]       	[expr ($MFloorMass5)/($NMBays)/8]             [expr {$mass_mom_joint/8}]
     mass    205041       [expr ($MFloorMass5)/($NMBays)/6]       	[expr ($MFloorMass5)/($NMBays)/6]             [expr {$mass_mom_joint/6}]
     mass    205042       [expr ($MFloorMass5)/($NMBays)/6]       	[expr ($MFloorMass5)/($NMBays)/6]             [expr {$mass_mom_joint/6}]
     mass    205043       [expr ($MFloorMass5)/($NMBays)/6]       	[expr ($MFloorMass5)/($NMBays)/6]             [expr {$mass_mom_joint/6}]
     mass    205044       [expr ($MFloorMass5)/($NMBays)/6]       	[expr ($MFloorMass5)/($NMBays)/6]             [expr {$mass_mom_joint/6}]	
 
     mass    204011       [expr ($MFloorMass4)/($NMBays+1)/6]      	[expr ($MFloorMass4)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    204012       [expr ($MFloorMass4)/($NMBays+1)/6]      	[expr ($MFloorMass4)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    204013       [expr ($MFloorMass4)/($NMBays+1)/6]      	[expr ($MFloorMass4)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    204014       [expr ($MFloorMass4)/($NMBays+1)/6]      	[expr ($MFloorMass4)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    204021       [expr ($MFloorMass4)/($NMBays+1)/8]       [expr ($MFloorMass4)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    204022       [expr ($MFloorMass4)/($NMBays+1)/8]       [expr ($MFloorMass4)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    204023       [expr ($MFloorMass4)/($NMBays+1)/8]       [expr ($MFloorMass4)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    204024       [expr ($MFloorMass4)/($NMBays+1)/8]       [expr ($MFloorMass4)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    204031       [expr ($MFloorMass4)/($NMBays+1)/8]       [expr ($MFloorMass4)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    204032       [expr ($MFloorMass4)/($NMBays+1)/8]       [expr ($MFloorMass4)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    204033       [expr ($MFloorMass4)/($NMBays+1)/8]       [expr ($MFloorMass4)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    204034       [expr ($MFloorMass4)/($NMBays+1)/8]       [expr ($MFloorMass4)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    204041       [expr ($MFloorMass4)/($NMBays+1)/6]      	[expr ($MFloorMass4)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    204042       [expr ($MFloorMass4)/($NMBays+1)/6]      	[expr ($MFloorMass4)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    204043       [expr ($MFloorMass4)/($NMBays+1)/6]      	[expr ($MFloorMass4)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    204044       [expr ($MFloorMass4)/($NMBays+1)/6]      	[expr ($MFloorMass4)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]	
 
     mass    203011       [expr ($MFloorMass3)/($NMBays+1)/6]      	[expr ($MFloorMass3)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    203012       [expr ($MFloorMass3)/($NMBays+1)/6]      	[expr ($MFloorMass3)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    203013       [expr ($MFloorMass3)/($NMBays+1)/6]      	[expr ($MFloorMass3)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    203014       [expr ($MFloorMass3)/($NMBays+1)/6]      	[expr ($MFloorMass3)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    203021       [expr ($MFloorMass3)/($NMBays+1)/8]       [expr ($MFloorMass3)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    203022       [expr ($MFloorMass3)/($NMBays+1)/8]       [expr ($MFloorMass3)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    203023       [expr ($MFloorMass3)/($NMBays+1)/8]       [expr ($MFloorMass3)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    203024       [expr ($MFloorMass3)/($NMBays+1)/8]       [expr ($MFloorMass3)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    203031       [expr ($MFloorMass3)/($NMBays+1)/8]       [expr ($MFloorMass3)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    203032       [expr ($MFloorMass3)/($NMBays+1)/8]       [expr ($MFloorMass3)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    203033       [expr ($MFloorMass3)/($NMBays+1)/8]       [expr ($MFloorMass3)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    203034       [expr ($MFloorMass3)/($NMBays+1)/8]       [expr ($MFloorMass3)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    203041       [expr ($MFloorMass3)/($NMBays+1)/6]      	[expr ($MFloorMass3)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    203042       [expr ($MFloorMass3)/($NMBays+1)/6]      	[expr ($MFloorMass3)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    203043       [expr ($MFloorMass3)/($NMBays+1)/6]      	[expr ($MFloorMass3)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    203044       [expr ($MFloorMass3)/($NMBays+1)/6]      	[expr ($MFloorMass3)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
 
     mass    202011       [expr ($MFloorMass2)/($NMBays+1)/6]      	[expr ($MFloorMass2)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    202012       [expr ($MFloorMass2)/($NMBays+1)/6]      	[expr ($MFloorMass2)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    202013       [expr ($MFloorMass2)/($NMBays+1)/6]      	[expr ($MFloorMass2)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    202014       [expr ($MFloorMass2)/($NMBays+1)/6]      	[expr ($MFloorMass2)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    202021       [expr ($MFloorMass2)/($NMBays+1)/8]       [expr ($MFloorMass2)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    202022       [expr ($MFloorMass2)/($NMBays+1)/8]       [expr ($MFloorMass2)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    202023       [expr ($MFloorMass2)/($NMBays+1)/8]       [expr ($MFloorMass2)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    202024       [expr ($MFloorMass2)/($NMBays+1)/8]       [expr ($MFloorMass2)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    202031       [expr ($MFloorMass2)/($NMBays+1)/8]       [expr ($MFloorMass2)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    202032       [expr ($MFloorMass2)/($NMBays+1)/8]       [expr ($MFloorMass2)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    202033       [expr ($MFloorMass2)/($NMBays+1)/8]       [expr ($MFloorMass2)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    202034       [expr ($MFloorMass2)/($NMBays+1)/8]       [expr ($MFloorMass2)/($NMBays+1)/8]           [expr {$mass_mom_joint/8}]
     mass    202041       [expr ($MFloorMass2)/($NMBays+1)/6]      	[expr ($MFloorMass2)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    202042       [expr ($MFloorMass2)/($NMBays+1)/6]      	[expr ($MFloorMass2)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    202043       [expr ($MFloorMass2)/($NMBays+1)/6]      	[expr ($MFloorMass2)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]
     mass    202044       [expr ($MFloorMass2)/($NMBays+1)/6]      	[expr ($MFloorMass2)/($NMBays+1)/6]           [expr {$mass_mom_joint/6}]


# RBS Nodes

mass 206026 [expr ($MFloorMass6)/($NMBays)/6]		[expr ($MFloorMass6)/($NMBays)/6]		[expr {$mass_mom_joint/6}];
mass 206036 [expr ($MFloorMass6)/($NMBays)/8]		[expr ($MFloorMass6)/($NMBays)/8]		[expr {$mass_mom_joint/8}];
mass 206038 [expr ($MFloorMass6)/($NMBays)/8]		[expr ($MFloorMass6)/($NMBays)/8]		[expr {$mass_mom_joint/8}];
mass 206048 [expr ($MFloorMass6)/($NMBays)/6]		[expr ($MFloorMass6)/($NMBays)/6]		[expr {$mass_mom_joint/6}];	 
mass 205026 [expr ($MFloorMass5)/($NMBays)/6]		[expr ($MFloorMass5)/($NMBays)/6]		[expr {$mass_mom_joint/6}];
mass 205036 [expr ($MFloorMass5)/($NMBays)/8]		[expr ($MFloorMass5)/($NMBays)/8]		[expr {$mass_mom_joint/8}];
mass 205038 [expr ($MFloorMass5)/($NMBays)/8]		[expr ($MFloorMass5)/($NMBays)/8]		[expr {$mass_mom_joint/8}];
mass 205048 [expr ($MFloorMass5)/($NMBays)/6]		[expr ($MFloorMass5)/($NMBays)/6]		[expr {$mass_mom_joint/6}];
mass 204016 [expr ($MFloorMass4)/($NMBays+1)/6]		[expr ($MFloorMass4)/($NMBays+1)/6]		[expr {$mass_mom_joint/6}];
mass 204026 [expr ($MFloorMass4)/($NMBays+1)/8]		[expr ($MFloorMass4)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 204028 [expr ($MFloorMass4)/($NMBays+1)/8]		[expr ($MFloorMass4)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 204036 [expr ($MFloorMass4)/($NMBays+1)/8]		[expr ($MFloorMass4)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 204038 [expr ($MFloorMass4)/($NMBays+1)/8]		[expr ($MFloorMass4)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 204048 [expr ($MFloorMass4)/($NMBays+1)/6]		[expr ($MFloorMass4)/($NMBays+1)/6]		[expr {$mass_mom_joint/6}];
mass 203016 [expr ($MFloorMass3)/($NMBays+1)/6]		[expr ($MFloorMass3)/($NMBays+1)/6]		[expr {$mass_mom_joint/6}];
mass 203026 [expr ($MFloorMass3)/($NMBays+1)/8]		[expr ($MFloorMass3)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 203028 [expr ($MFloorMass3)/($NMBays+1)/8]		[expr ($MFloorMass3)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 203036 [expr ($MFloorMass3)/($NMBays+1)/8]		[expr ($MFloorMass3)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 203038 [expr ($MFloorMass3)/($NMBays+1)/8]		[expr ($MFloorMass3)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 203048 [expr ($MFloorMass3)/($NMBays+1)/6]		[expr ($MFloorMass3)/($NMBays+1)/6]		[expr {$mass_mom_joint/6}];
mass 202016 [expr ($MFloorMass2)/($NMBays+1)/6]		[expr ($MFloorMass2)/($NMBays+1)/6]		[expr {$mass_mom_joint/6}];
mass 202026 [expr ($MFloorMass2)/($NMBays+1)/8]		[expr ($MFloorMass2)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 202028 [expr ($MFloorMass2)/($NMBays+1)/8]		[expr ($MFloorMass2)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 202036 [expr ($MFloorMass2)/($NMBays+1)/8]		[expr ($MFloorMass2)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 202038 [expr ($MFloorMass2)/($NMBays+1)/8]		[expr ($MFloorMass2)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 202048 [expr ($MFloorMass2)/($NMBays+1)/6]		[expr ($MFloorMass2)/($NMBays+1)/6]		[expr {$mass_mom_joint/6}];

mass 206027 [expr ($MFloorMass6)/($NMBays)/6]		[expr ($MFloorMass6)/($NMBays)/6]		[expr {$mass_mom_joint/6}];
mass 206037 [expr ($MFloorMass6)/($NMBays)/8]		[expr ($MFloorMass6)/($NMBays)/8]		[expr {$mass_mom_joint/8}];
mass 206039 [expr ($MFloorMass6)/($NMBays)/8]		[expr ($MFloorMass6)/($NMBays)/8]		[expr {$mass_mom_joint/8}];
mass 206049 [expr ($MFloorMass6)/($NMBays)/6]		[expr ($MFloorMass6)/($NMBays)/6]		[expr {$mass_mom_joint/6}];
mass 205027 [expr ($MFloorMass5)/($NMBays)/6]		[expr ($MFloorMass5)/($NMBays)/6]		[expr {$mass_mom_joint/6}];
mass 205037 [expr ($MFloorMass5)/($NMBays)/8]		[expr ($MFloorMass5)/($NMBays)/8]		[expr {$mass_mom_joint/8}];
mass 205039 [expr ($MFloorMass5)/($NMBays)/8]		[expr ($MFloorMass5)/($NMBays)/8]		[expr {$mass_mom_joint/8}];
mass 205049 [expr ($MFloorMass5)/($NMBays)/6]		[expr ($MFloorMass5)/($NMBays)/6]		[expr {$mass_mom_joint/6}];
mass 204017 [expr ($MFloorMass4)/($NMBays+1)/6]		[expr ($MFloorMass4)/($NMBays+1)/6]		[expr {$mass_mom_joint/6}];
mass 204027 [expr ($MFloorMass4)/($NMBays+1)/8]		[expr ($MFloorMass4)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 204029 [expr ($MFloorMass4)/($NMBays+1)/8]		[expr ($MFloorMass4)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 204037 [expr ($MFloorMass4)/($NMBays+1)/8]		[expr ($MFloorMass4)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 204039 [expr ($MFloorMass4)/($NMBays+1)/8]		[expr ($MFloorMass4)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 204049 [expr ($MFloorMass4)/($NMBays+1)/6]		[expr ($MFloorMass4)/($NMBays+1)/6]		[expr {$mass_mom_joint/6}];
mass 203017 [expr ($MFloorMass3)/($NMBays+1)/6]		[expr ($MFloorMass3)/($NMBays+1)/6]		[expr {$mass_mom_joint/6}];
mass 203027 [expr ($MFloorMass3)/($NMBays+1)/8]		[expr ($MFloorMass3)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 203029 [expr ($MFloorMass3)/($NMBays+1)/8]		[expr ($MFloorMass3)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 203037 [expr ($MFloorMass3)/($NMBays+1)/8]		[expr ($MFloorMass3)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 203039 [expr ($MFloorMass3)/($NMBays+1)/8]		[expr ($MFloorMass3)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 203049 [expr ($MFloorMass3)/($NMBays+1)/6]		[expr ($MFloorMass3)/($NMBays+1)/6]		[expr {$mass_mom_joint/6}];
mass 202017 [expr ($MFloorMass2)/($NMBays+1)/6]		[expr ($MFloorMass2)/($NMBays+1)/6]		[expr {$mass_mom_joint/6}];
mass 202027 [expr ($MFloorMass2)/($NMBays+1)/8]		[expr ($MFloorMass2)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 202029 [expr ($MFloorMass2)/($NMBays+1)/8]		[expr ($MFloorMass2)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 202037 [expr ($MFloorMass2)/($NMBays+1)/8]		[expr ($MFloorMass2)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 202039 [expr ($MFloorMass2)/($NMBays+1)/8]		[expr ($MFloorMass2)/($NMBays+1)/8]		[expr {$mass_mom_joint/8}];
mass 202049 [expr ($MFloorMass2)/($NMBays+1)/6]		[expr ($MFloorMass2)/($NMBays+1)/6]		[expr {$mass_mom_joint/6}];

 # Check: Total mass applied to building is: 35.8611111111111.  This was summed in the VB loop when mass was applied. This does not include small masses applied for convergence. 
 		
##############################################################################################################################
#          	 									Display Model with Node Numbers	and EigenValueAnalysis									 	     #
##############################################################################################################################
 
# display the model with the node numbers
	# source DisplayModel2D.tcl
	# source DisplayPlane.tcl
	# DisplayModel2D NodeNumbers
	# source EigenValueAnalysis.tcl;
	
#########################################################################################################################
### DEFINE DAMPING OBJECTS
 
# Define damping for all beams and all columns (i.e. elastic elements), but not on the joints b/c they have the nonlinearity in them.  This is the approach proposed by Medina.  Compute the damping paramters based on Chopra text page 457.
	  set lambdalist [eigen -fullGenLapack 3];
	  set omegaI [expr {sqrt([lindex $lambdalist 0])}];
	  set omegaJ [expr {sqrt([lindex $lambdalist 2])}];
      # set omegaI [expr (2.0 * $pi) / $periodForRayleighDamping_1]
      # set omegaJ [expr (2.0 * $pi) / ($periodForRayleighDamping_2)]
      set alpha1Coeff [expr (2.0 * $omegaI * $omegaJ) / ($omegaI + $omegaJ)]
      set alpha2Coeff [expr (2.0) / ($omegaI + $omegaJ)]
      set alpha1  [expr $alpha1Coeff * $dampRat * $dampRatF]
      set alpha2  [expr $alpha2Coeff * $dampRat * $dampRatF]
      set alpha2ToUse [expr 1.1 * $alpha2];   # 1.1 factor is becuase we apply to only LE elements
      region 1 -eleRange  20000   39999   -rayleigh $alpha1 0 $alpha2ToUse 0;    # Initial stiffness"
	  region 2 -nodeRange  200000   299999   -rayleigh $alpha1 0 $alpha2ToUse 0;   # Initial stiffness"
 
#########################################################################################################################

########## Define Additional Information ##########

# Define the number of stories in the frame
set num_stories 5

# Define the control nodes used to compute story drifts
set ctrl_nodes {
    3013
    3021
    3031
    3041
    3051
    3061
}
#########################################################################################################################

puts "Model Built!"

# source [file join [file dirname [info script]] display2D.tcl]
source [file join [file dirname [info script]] PerformGravityLoadAnalysis.tcl]

puts "Gravity analysis is Done!"
# source DefineAllRecorders.tcl
	  
# # DisplayPlane
	# set window_width 650
	# set window_height [expr ($window_width*633)/1348]
	# recorder display "BaselineModel" 600 20 $window_width $window_height -wipe
	# DisplayPlane DeformedShape 1.0 XY 0 0
	# vup 0 1 0
	# vpn 0 0 1
	# prp 0 0 400

# # Ground motion scale factor
# set scalefactor [expr $g*$scale/100*$MCE_SF]

# # Run Dynamic Anallysis.tcl
# source DefineTimeHistory1.tcl

