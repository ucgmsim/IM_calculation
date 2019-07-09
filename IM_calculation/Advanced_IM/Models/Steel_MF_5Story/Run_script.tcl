# Run-script for Response History Analysis
#  

if { $argc != 3 } {
    puts "The add.tcl script requires arguments."
    puts "For example, tclsh add.tcl event_name".
    puts "Please try again."
} else {
    set GM_name [lindex $argv 0]
    # puts $GM_name
    set FEMs [lindex $argv 1]	
    # puts $FEMs	
    set dir [lindex $argv 2]	
    # puts $dir	 			
   }

puts "Start Analyzing..."
set model_name $FEMs
set GM_name $GM_name
set station_num 1
# for {set iRec 1} {$iRec <= $st_number} {incr iRec} {
# }

#find all GMs and loop through them
  foreach gMotion [glob -nocomplain -directory ../GMotions3/$GM_name *.$dir] {     
  
         set station_path [string range $gMotion 0 end-0]
         set station_name [string range $gMotion 30 end-4]	
         # puts "station_path = $station_path"		 
	 puts "GM=$GM_name"	
	 puts "stations number = $station_num"
         puts "FEMs=$FEMs"
         puts " "
	 puts "-------------------------------"
	 puts " "
	 
       	source Models/general/GMs.tcl
	# puts "Station no= $iRec"	
	# puts "Type= $type_GM"
	puts " "
	source Models/$FEMs/Original/model.tcl
	source Models/general/Period.tcl
	source Models/$FEMs/Recorders.tcl
	source Models/general/Rha.tcl
	
	puts " "
	puts "---------------------------------------------------------------------------"
	puts " "
	incr station_num				
    wipe
	          
}	