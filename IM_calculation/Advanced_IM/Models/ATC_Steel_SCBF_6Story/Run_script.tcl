if { $argc != 2 } {
    puts "The add.tcl script requires arguments."
    puts "For example, tclsh add.tcl event_name".
    puts "Please try again."
} else {
    set GM_path [lindex $argv 0]
    
    set Output_path [lindex $argv 1]	
    puts "Output_path = $Output_path" 		
    			
    }

puts "Start Analyzing..."	


set FEMs SCBF_6Story


    puts "FEMs=$FEMs"
    puts " "
	puts "-------------------------------"
	puts " "
	puts [file join [file dirname [info script]]]
	puts "-------------------------------"
	 
    source [file join [file dirname [info script]] ../general/GMs.tcl]	

	puts " "
	source [file join [file dirname [info script]] Original/Models/$FEMs/build_modified_atc84_model.tcl]
	source [file join [file dirname [info script]] ../general/Period.tcl]
	source [file join [file dirname [info script]] Recorders.tcl]
	source [file join [file dirname [info script]] ../general/Rha.tcl]
	
	puts " "
	puts "---------------------------------------------------------------------------"
	puts " "
	
    wipe
	         

