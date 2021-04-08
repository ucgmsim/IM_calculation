# -----------------------------------------------------
#
# ---Nonlinear Analysis / central differene method --- 
#
# -----------------------------------------------------


set Outputs_cd [open [file join [file dirname [info script]] $Output_path/Analysis_CD_${st1}.txt] w]

set pi 3.14
set Tmax $Tmax1

# Define the time step used to run the analysis using the central difference scheme
set dt_factor 0.8
set periods_file [open [file join [file dirname [info script]] ../$FEMs/Original/Model_Info/periods.out]]
while {[gets $periods_file line] >= 0} {set period $line}
close $periods_file
set dt_cd [expr {$dt_factor*$period/$pi}]

constraints Transformation
numberer RCM
system SparseGEN
algorithm Linear
integrator CentralDifference
analysis Transient

set nSteps_cd [expr int($Tmax/$dt_cd)+1]


puts "------------------------ Trying: central differnce method, dt=$dt_cd --------------------------"
set ok [analyze $nSteps_cd $dt_cd]
set curTime [getTime]
set DT [expr $Tmax-$curTime]
set iTry 1

if {$DT > $dt_cd} {
    puts ""
	puts "!!!!!!!!!!!!!!!!!!!!!!!!!!! Central differene Failed !!!!!!!!!!!!!!!!!!!!!!!!!!"
	puts ""
	lappend status_cd   "Failed" 
	puts $Outputs_cd $status_cd 
	flush $Outputs_cd	
	
} else {
	puts "-------------------------- Central differene Successful! ---------------------------"
	puts ""
	lappend status_cd  "Successful" 
	puts $Outputs_cd $status_cd 
	flush $Outputs_cd
    puts "                           endTime= [getTime]                              "
    puts ""
    puts "--------------------Response history analysis is Done!---------------------"

}
close $Outputs_cd
