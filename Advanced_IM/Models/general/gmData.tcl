#
# Procedure for reading and writing GM time step (dt), total Time (Tmax) 
# and Station name from GM txt file 
#
proc gmData {gmFile outFile dt1 Tmax1 st1} {
	
	upvar $dt1 dt 
	upvar $Tmax1 Tmax
	upvar $st1 st
	set input [open $gmFile r]
	set output [open $outFile w]
	set lineList [split [read $input] \n]
	set n 0
	foreach line $lineList {
		incr n
		if {$n==1} {
		    set st [lindex $line 0]
		    puts "Station= $st"
		   }
		    incr i	
		if {$n == 2} {
			set dt [lindex $line 1]
			puts "dt= $dt"
			set npnts [lindex $line 0]
			# puts "npnts= $npnts"
			set Tmax [expr $npnts*$dt]
			puts "Tmax= $Tmax"
			
		}
		if {$n > 2} {
			puts $output $line
		}
	}
	close $input
        close $output
}

