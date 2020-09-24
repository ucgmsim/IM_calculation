#########################################################################
#     please keep this notification at the beginning of this file       #
#                                                                       #
# Code to perform pushover analysis by avoidance of numerical divergence#
#                                                                       #
#                  Developed by Seyed Alireza Jalali                    #
#        as part of the OpenSees course in civil808 institute           #
#  for more information and any questions about this code,              #
#               Join  "OpenSees SAJ" telegram group:                    #
#            (https://t.me/joinchat/CJlXoECQvxiJXal0PkLfwg)             #
#                     or visit: www.civil808.com                        #
#                                                                       #
#      DISTRIBUTION OF THIS CODE WITHOUT WRITTEN PERMISSION FROM        #
#                THE DEVELOPER IS HEREBY RESTRICTED                     #
#########################################################################

#source this file after building the model and applying the pushover load pattern

set incr 0.01										;#maximum displacement increment in units of length
													;#the algorith may resolve this value to enhance convergence
set LBuilding $roof_height							;#the total height of the building in units of length	
										
# set targetDriftList "0.0035 0. -0.0035 0."		;#for cyclic analysis
set targetDriftList "$target_roof_drift"			        ;#for monotonic analysis
set tol 1.e-4										;#minimum tolerance of convergence test
													;#the algorithm may decrease increase this value  to enhance convergence
set algoList "{NewtonLineSearch 0.65} ModifiedNewton KrylovNewton"
													;#the desired list of algorithms; broyden and BFGS may lead to unacceptable
													;#values in static analysis
set roofNode $roof_node									        ;#tag of the control node (roof node)
set logfileId [open log.txt w+]						;#a file in which information about the analysis procedure are printed

set incr1 $incr
set tol1 $tol
set numAlgos [llength $algoList]
constraints Transformation
numberer RCM
system UmfPack
test NormDispIncr $tol 100
analysis Static
set failureFlag 0
set endDisp 0
for {set iDrift 0} {$iDrift < [llength $targetDriftList]} {incr iDrift} {
	set targetDrift [lindex $targetDriftList $iDrift]
	set targetDisp [expr $targetDrift*$LBuilding]
	puts "***************** Applying targetDrift= $targetDrift, targetDisp= $targetDisp ****************"
	puts $logfileId "***************** Applying targetDrift= $targetDrift, targetDisp= $targetDisp ****************"
	set curD [nodeDisp $roofNode 1]
	set deltaD [expr $targetDisp - $curD]
	set nSteps [expr int(abs($deltaD)/$incr1)]
	algorithm Newton
	integrator DisplacementControl $roofNode 1 [expr abs($deltaD)/$deltaD*$incr]
	puts "########################## Trying: Newton, incr=$incr1 ##########################"
	puts $logfileId "########################## Trying: Newton, incr=$incr1 ##########################"
	set ok [analyze $nSteps]
	set curD [nodeDisp $roofNode 1]
	set deltaD [expr $targetDisp-$curD]
	set iTry 1
	while {[expr abs($deltaD)] > $incr} {
		puts "~~~~~~~~~~~~~~~~~~~~~~~~~~ curD= $curD, deltaD= $deltaD ~~~~~~~~~~~~~~~~~~~~~~~~~~"
		puts $logfileId "~~~~~~~~~~~~~~~~~~~~~~~~~~ curD= $curD, deltaD= $deltaD ~~~~~~~~~~~~~~~~~~~~~~~~~~"
		integrator DisplacementControl $roofNode 1 [expr abs($deltaD)/$deltaD*$incr1]
		if {$iTry <= $numAlgos} {
			set algo [lindex $algoList [expr $iTry-1]]
			puts "########################## Trying: [lindex $algo 0], incr=$incr1 ##########################"
			puts $logfileId "########################## Trying: [lindex $algo 0], incr=$incr1 ##########################"
			test NormDispIncr $tol1 30
			eval "algorithm $algo"
			set nSteps [expr int(10.*$incr/$incr1)]
			set ok [analyze $nSteps]
			if {$ok == 0} {
				set curD [nodeDisp $roofNode 1]
				set deltaD [expr $targetDisp-$curD]
				set nSteps [expr int(abs($deltaD)/$incr)]
				set ok [analyze $nSteps $incr]
				set incr1 $incr
				set tol1 $tol
				set iTry 0
			}
		} else {
			set iTry 0
			set incr1 [expr $incr1/3.]
			set tol1 [expr $tol1*3.]
			if {[expr $incr1/$incr] < 1.e-3} {
				set failureFlag 1
				break
			}
		}
		incr iTry
		set curD [nodeDisp $roofNode 1]
		set deltaD [expr $targetDisp-$curD]		
	}
	set endDisp [nodeDisp $roofNode 1]
	if {$failureFlag == 0} {
		puts "########################## Analysis Successful! ##########################"
		puts $logfileId "########################## Analysis Successful! ##########################"
	} else {
		puts "!!!!!!!!!!!!!!!!!!!!!!!!!! Analysis Interrupted !!!!!!!!!!!!!!!!!!!!!!!!!!"
		puts $logfileId "!!!!!!!!!!!!!!!!!!!!!!!!!! Analysis Interrupted !!!!!!!!!!!!!!!!!!!!!!!!!!"
	}
	puts "~~~~~~~~~~~~~~~~~~~~~~~~~~ endDisp= $endDisp ~~~~~~~~~~~~~~~~~~~~~~~~~~"
	puts $logfileId "~~~~~~~~~~~~~~~~~~~~~~~~~~ endDisp= $endDisp ~~~~~~~~~~~~~~~~~~~~~~~~~~"
}
