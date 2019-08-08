# ----------------------------------------------
#
# ---Nonlinear Analysis--- 
#      One algortithm 

# constraints Transformation
# numberer RCM
# system UmfPack
# test NormUnbalance 1.e-6 100
# algorithm Newton
# integrator Newmark 0.5 0.25
# analysis Transient
# set nSteps [expr int($Tmax1/$dt1)+1]
# analyze $nSteps $dt 

# ----------------------------------------------
# source getMaxResp.tcl
set Tmax $Tmax1
set dt $dt1

set Outputs_nm [open [file join [file dirname [info script]] $Output_path/Analysis_NM_${st1}.txt] w]
set Outputs_cd [open [file join [file dirname [info script]] $Output_path/Analysis_CD_${st1}.txt] w]

set algoList "{NewtonLineSearch 0.65} ModifiedNewton KrylovNewton BFGS Broyden"

constraints Transformation
numberer RCM
system UmfPack
test NormUnbalance 1.e-6 100
algorithm Newton
integrator Newmark 0.5 0.25
analysis Transient
set failureFlag 0
set endTime 0
set nSteps [expr int($Tmax/$dt1)+1]


puts "------------------------ Trying: Newton, dt=$dt1 --------------------------"
set ok [analyze $nSteps $dt1]
set curTime [getTime]
set DT [expr $Tmax-$curTime]
set iTry 1
while {$DT > $dt} {
	# if {[getMaxResp recTags] > 0.1} {
		# set failureFlag 1
		# break
	# }
	puts "~~~~~~~~~~~~~~~~~~~~~~~~~~ curTime= $curTime, DT= $DT ~~~~~~~~~~~~~~~~~~~~~~~~~~"
	if {$iTry <= 5} {
		set algo [lindex $algoList [expr $iTry-1]]
		puts "------------------------ Trying: [lindex $algo 0], dt=$dt1 --------------------------"
		puts ""
		eval "algorithm $algo"
		set nSteps [expr int(0.1/$dt1)]
		set ok [analyze $nSteps $dt1]
		if {$ok == 0} {
			set curTime [getTime]
			set DT [expr $Tmax-$curTime]
			set nSteps [expr int($DT/$dt)]
			set ok [analyze $nSteps $dt]
			set dt1 $dt
			set iTry 0
		}
	} else {
		set iTry 0
		set dt1 [expr $dt1/10]
		if {[expr $dt1/$dt] < 1.e-3} {
			set failureFlag 1
			break
		}
	}
	incr iTry
	set curTime [getTime]
	set DT [expr $Tmax-$curTime]		
}
set endTime [getTime]
if {$failureFlag == 0} {
        puts ""
	puts "-------------------------- Newmark Successful! ---------------------------"
	puts ""
	lappend status "Successful" 
	puts $Outputs_nm $status 
	flush $Outputs_nm	
	
} else {
	puts "!!!!!!!!!!!!!!!!!!!!!!!!!!! Newmark Failed !!!!!!!!!!!!!!!!!!!!!!!!!!"
	puts ""
	lappend status  "Failed" 
	puts $Outputs_nm $status 
	flush $Outputs_nm
	
	source [file join [file dirname [info script]] Rha_central_difference.tcl]
}
close $Outputs_nm	
close $Outputs_cd	

        puts "                           endTime= [getTime]                              "
	puts ""
        puts "--------------------Response history analysis is Done!---------------------" 