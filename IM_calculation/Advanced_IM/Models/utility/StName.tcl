# a code to extract the station names from the station list (Name, dt, Tmax)
#
set stname [open ../SAC_Steel_MFs/report/station_list.txt r]
set outst [open ../SAC_Steel_MFs/report/station_name.txt w]
set lineList [split [read $stname]]

	for {set x 1} {$x<=[expr 55*3]} { set x [expr {$x+3}]} {
	        set st [lindex $lineList [expr {$x - 1}]]
	         puts  $outst $st
	}
close $stname
close $outst