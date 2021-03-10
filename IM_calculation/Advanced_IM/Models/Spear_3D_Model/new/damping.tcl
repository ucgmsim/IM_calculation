# define damping

# Set percentage of critical damping and two mode numbers to apply it to. 
set damping 0.05
set modeI 1
set modeJ 2

set omega2List [eigen 10]

#apply rayleigh damping
set omegaI [expr sqrt([lindex $omega2List [expr $modeI-1]])]
set omegaJ [expr sqrt([lindex $omega2List [expr $modeJ-1]])]
set alphaM [expr $damping*(2.*$omegaI*$omegaJ)/($omegaI+$omegaJ)];	# M-prop. damping; D = alphaM*M
set betaK [expr 2.*$damping/($omegaI+$omegaJ)];         		# current-K;      +beatKcurr*KCurrent

# Apply damping to only the linear elastic frame elements, excluding all plastic hinges 
#  Use the initial stiffness matrix at each step to form the damping matrix.
 region 1 -eleRange  1   27   -rayleigh $alphaM 0 $betaK 0;    # Initial stiffness"       for columns 
 region 1 -eleRange  101   1403   -rayleigh $alphaM 0 $betaK 0;    # Initial stiffness"       for beams  
 
 puts "Damping object was defined!"
