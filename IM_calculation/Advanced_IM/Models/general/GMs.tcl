# Reading GMs files and preparing for the analysis	
        
	source [file join [file dirname [info script]] gmData.tcl]
		
	gmData "$GM_path" $Output_path dt1 Tmax1 st1 outfile comp
	
	set station_name $st1
	set station [open $Output_path/station_property.txt w]
	lappend list $st1 $comp $dt1 $Tmax1
	 puts $station $list 
	close $station
	
