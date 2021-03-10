# Reading GMs files and preparing for the analysis	
set i 0
 foreach	path  $GM_path {
         set path [lindex $GM_path $i]
         puts "path=$path"  
	source [file join [file dirname [info script]] gmData.tcl]
	
	# gmData "C:/Users/vlo23/Automated_Workflow/OpenSees_Models_Ver5/GMotions3/ChCh22Feb2011_Sim/ADCS.000" "C:/Users/vlo23/Automated_Workflow/OpenSees_Models_Ver5/GMotions3/ChCh22Feb2011_Sim/transformed/ADCS.txt" dt1 Tmax1 st1
	 set com [lindex $dir $i]
	 puts $com
	 file mkdir $Output_path/$com
	gmData $path $Output_path/$com dt1 Tmax1 st1 outfile comp
	
	lappend outfile_path $outfile
	set station_name $st1
	set station [open [file join [file dirname [info script]] $Output_path/$com/station_property.txt ] w]
	lappend list $st1 $comp $dt1 $Tmax1
	 puts $station $list 
	close $station
	unset list
	incr i
		
}	