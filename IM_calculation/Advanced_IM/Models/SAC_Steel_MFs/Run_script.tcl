# set modelpath Original/Models
# puts  "$modelpath"
# set modeldirs [lsort [glob -directory ../Models/$FEMs/$modelpath -type d *Story]]
# puts $modeldirs
# set lineList [split $modeldirs \n]
# puts "$lineList"

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
    puts $dir	 			
    }

puts "Start Analyzing..."
set modeldirs "3Story 9Story"
set model_name $FEMs
set GM_name $GM_name


foreach modeldir $modeldirs {
         set station_num 1
    # for {set iRec 1} {$iRec <= $st_number} {incr iRec} {
    # }	
	 foreach gMotion [glob -nocomplain -directory ../GMotions3/$GM_name *.$dir] {     
         set station_path [string range $gMotion 0 end-0]
         set station_name [string range $gMotion 30 end-4]  
    
         puts "GM=$GM_name"	
	 puts "stations number = $station_num"
         puts "FEMs=$FEMs"
         puts "Model= $modeldir"		 
         puts " "
	 puts "-------------------------------"
	 puts " "  	
        source Models/general/GMs.tcl	
	# puts "Station no= $iRec"	
	# puts "Type= $type_GM"
	puts " "
	
	source Models/$FEMs/Original/constants_units_kip_in.tcl	
        source Models/$FEMs/Original/create_steel_mf_model.tcl
	source Models/$FEMs/Original/models/$modeldir/frame_data.tcl	
	CreateSteelMFModel Models/$FEMs/Original/models/$modeldir/frame_data.tcl $FEMs
	 source Models/general/Period.tcl
		
	# puts $modeldir
	source Models/$FEMs/Recorders.tcl
	source Models/general/Rha.tcl
	
	puts " "
	puts "---------------------------------------------------------------------------"
	puts " "
	incr station_num	
	# Wipe previous model definitions and build the model
    wipe
    # Display the model name
    # puts \n$modeldir
    }		
}