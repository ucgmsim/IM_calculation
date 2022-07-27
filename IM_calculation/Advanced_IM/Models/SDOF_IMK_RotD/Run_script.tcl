if { $argc != 3 } {
    puts "The add.tcl script requires arguments."
    puts "For example, tclsh add.tcl event_name".
    puts "Please try again."
} else {
    set GM_path_0 [lindex $argv 0]
    puts "GM_path_0 = $GM_path_0"	
    set GM_path_90 [lindex $argv 1]
    puts "GM_path_90 = $GM_path_90"		
    set Output_path [lindex $argv 2]	
    puts "Output_path = $Output_path" 		
    		
   } 

puts "-----"

set GM_path "{$GM_path_0} {$GM_path_90}"
set dir "000 090"	
set rot_max ""

puts "Start Analyzing..."

set run_flag 0
set FEMs SDOF_IMK_rotd 

    puts "FEMs=$FEMs"
    puts " "
	puts "-------------------------------"
	puts " "  	
    puts " "
	puts [file join [file dirname [info script]]]
	puts "-------------------------------"
	
proc load_tcls {run_flag GM_path GM_path_0 GM_path_90 Output_path dir FEMs thetaRad dt Tmax st} {	

	puts "run_flag = $run_flag"
	upvar $Tmax Tmax1
    upvar $dt dt1	
    upvar $st st1		 
	source [file join [file dirname [info script]] extra_rotd/GMs.tcl]	
	
	source [file join [file dirname [info script]] Original/Nonlinear_SDOF_IMK.tcl]	
    set scale 1.0
	
	source [file join [file dirname [info script]] Recorders.tcl]
}

set Pi 3.14 ;

for {set theta 0} {$theta < 181} {incr theta} {
set thetaRad [expr $theta*$Pi/180.0] 
puts "theta = $thetaRad"

load_tcls $run_flag $GM_path $GM_path_0 $GM_path_90 $Output_path $dir $FEMs $thetaRad dt Tmax st
set Tmax1 $Tmax
set dt1 $dt
set st1 $st
source [file join [file dirname [info script]] ../general/Rha.tcl]



if {$run_flag ==1} {
	wipe
	load_tcls $run_flag "$GM_path" $Output_path $FEMs dt Tmax st
	set Tmax1 $Tmax
	set dt1 $dt
	set st1 $st
	source [file join [file dirname [info script]] ../general/Rha_central_difference.tcl]
	
	puts " "
	puts "---------------------------------------------------------------------------"
	puts " "
}	
wipe

set Output_path_resp "$Output_path/rotd/env_disp/disp.out"
source [file join [file dirname [info script]] extra_rotd/MaxResp.tcl]

lappend rot_max [MaxResp $Output_path_resp]
# puts "rot_max = $rot_max"

set outFile [file join  $Output_path "max_rotD_resp.txt"]
set output [open $outFile w]
puts $output $rot_max
close $output
puts " "
puts "---------------------------------------------------------------------------"
puts " "
}

