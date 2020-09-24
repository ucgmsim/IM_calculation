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

set GM_path "{$GM_path_0} {$GM_path_90}"
set dir "000 090"	
set g 9.81
puts "Start Analyzing..."


set FEMs Spear_3D_Model 

         puts "FEMs=$FEMs"
        	 
         puts " "
	 puts "-------------------------------"
	 puts " "  	
         source [file join [file dirname [info script]] extra_3D/GMs.tcl]	
	
	puts " "
	puts [file join [file dirname [info script]]]
	puts "-------------------------------"
	
	source [file join [file dirname [info script]] Original/vozlisca.tcl]	
    
    puts "-------------------------------break point 1"
    
    source [file join [file dirname [info script]] Original/ovojniceZL.tcl]
    
    puts "-------------------------------break point 2"
	source [file join [file dirname [info script]] Original/elementiZL.tcl]	
    puts "-------------------------------break point 3"
	source [file join [file dirname [info script]] Original/Gravity.tcl]	
    puts "-------------------------------break point 4"
	source [file join [file dirname [info script]] ../general/Period.tcl]
    source [file join [file dirname [info script]] extra_3D/damping.tcl]
		
    puts "-------------------------------break point 5"
	source [file join [file dirname [info script]] extra_3D/time_series.tcl]
    puts "-------------------------------break point 6"
	source [file join [file dirname [info script]] Recorders.tcl]
    puts "-------------------------------break point 7"
	source [file join [file dirname [info script]] extra_3D/Rha.tcl]
	
	puts " "
	puts "---------------------------------------------------------------------------"
	puts " "
	
    wipe
    
