# Run-script for Response History Analysis
#  

if { $argc != 2 } {
    puts "The add.tcl script requires arguments."
    puts "For example, tclsh add.tcl event_name".
    puts "Please try again."
} else {
    set GM_path [lindex $argv 0]
    puts "GM_path = $GM_path"
    puts " "	
    set Output_path [lindex $argv 1]	
    puts "Output_path = $Output_path" 			
   }   

puts " "
puts "Start Analyzing..."
set FEMs Steel_MF_5Story 

puts " "
puts "FEMs = $FEMs"
puts " "
puts "-------------------------------"
puts " "

source [file join [file dirname [info script]] ../general/GMs.tcl]

puts " "
source [file join [file dirname [info script]] Original/Model.tcl]
source [file join [file dirname [info script]] ../general/Period.tcl]
source [file join [file dirname [info script]] ../$FEMs/Recorders.tcl]
source [file join [file dirname [info script]] ../general/Rha.tcl]

puts " "
puts "---------------------------------------------------------------------------"
puts " "
		
wipe
	          
