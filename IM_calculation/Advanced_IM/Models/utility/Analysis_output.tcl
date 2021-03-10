# a code to put the analysis status in order
#
set GM_name $GM_name;             	 #  

set input [open ../Steel_MF_5Story/report/Analysis_out_${GM_name}.txt r]
set output [open ../Steel_MF_5Story/report/Analysis_status_${GM_name}.txt w]
set lineList [split [read $input]]

set n 1

for {set x 1} {$x<=[expr 40*3]} { set x [expr {$x+1}]} {
      
	   
	 set q [expr $x/3] 	   
	 set r [expr $x-3*$q] 	 
        	  
	  if {$r == 1} {
	   set res [lindex $lineList [expr {$x - 1}]]
           puts  $output $res	   
	  } elseif {$r == 2} {
	   set j1 [lindex $lineList [expr {$x - 1}]]
	   } else {set j2 [lindex $lineList [expr {$x - 1}]]
           puts $output "$j1 $j2"
	   puts $output ""	   
           }
	     
}

	# for {set x 1} {$x<=[expr 40*3]} { set x [expr {$x+1}]} {
	        # set res [lindex $lineList [expr {$x - 1}]]
	         # puts  $output $res
	# }
close $input
close $output