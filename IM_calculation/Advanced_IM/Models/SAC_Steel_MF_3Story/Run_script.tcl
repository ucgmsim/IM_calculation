if { $argc != 2 } {
    puts "The add.tcl script requires arguments."
    puts "For example, tclsh add.tcl event_name".
    puts "Please try again."
} else {
    set GM_path [lindex $argv 0]
     puts "GM_path = $GM_path"
    
    set Output_path [lindex $argv 1]    
    puts "Output_path = $Output_path"           
                
    }

puts "Start Analyzing..."

set run_flag 0
set FEMs SAC_Steel_MF_3Story 

puts "FEMs=$FEMs"
puts " "
puts "-------------------------------"
puts " "
puts " "
puts [file join [file dirname [info script]]]
puts "-------------------------------"

proc load_tcls {run_flag GM_path Output_path FEMs dt Tmax st} { 

    puts "run_flag = $run_flag"
    upvar $Tmax Tmax1
    upvar $dt dt1  
    upvar $st st1    
    source [file join [file dirname [info script]] ../general/GMs.tcl]
    source [file join [file dirname [info script]] Original/constants_units_kip_in.tcl] 
    source [file join [file dirname [info script]] Original/create_steel_mf_model.tcl]
    source [file join [file dirname [info script]] Original/models/frame_data.tcl]  
    CreateSteelMFModel [file join [file dirname [info script]] Original/models/frame_data.tcl] $FEMs $Output_path
    source [file join [file dirname [info script]] ../general/Period.tcl]
    source [file join [file dirname [info script]] Recorders.tcl]
}

load_tcls $run_flag "$GM_path" $Output_path $FEMs dt Tmax st
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

