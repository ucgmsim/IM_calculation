# Reading GMs files and preparing for the analysis	
	source Models/general/gmData.tcl
	# gmData GMfiles/$type/$iRec.000 GMfiles/$type/transformed/$iRec.txt dt1 Tmax1 st1
	gmData $station_path ../GMotions3/$GM_name/transformed/$station_name.txt dt1 Tmax1 st1
	set station [open Models/$FEMs/report/station_list.txt w]
	lappend list $st1 $dt1 $Tmax1
	 puts $station $list 
	close $station
	