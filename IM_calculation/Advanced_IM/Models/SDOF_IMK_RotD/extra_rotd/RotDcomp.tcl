#
# Procedure for reading two components and calculate the rotD component
#

proc RotDcomp {Output_path filePathX filePathY thetaRad} { 
    
	set scale_1 cos($thetaRad)
	set scale_2 sin($thetaRad)
	
	# upvar $outfile outFile
	set input_X [open $filePathX]
	set list_X [split [read $input_X] "\n"]
	# puts $list_1
	foreach rec_x $list_X {

        ## Split into fields on space
        set fields_x [split $rec_x "\S+"]
		
		foreach i $fields_x {
			set ii [regexp -all -inline {\S+} $i]
			# puts "ii=$ii"
			
			foreach iii $ii {
				set comp_x [expr $iii * $scale_1]
				# puts $iii
				# puts $comp_x
				lappend comp_00 $comp_x
				
			}
		}
	}
	
	
	set input_Y [open $filePathY]
	set list_Y [split [read $input_Y] "\n"]
	# puts $list_1
	foreach rec_y $list_Y {

        ## Split into fields on space
        set fields_y [split $rec_y "\S+"]
		
		foreach j $fields_y {
			set jj [regexp -all -inline {\S+} $j]
			# puts "jj=$jj"
			
			foreach jjj $jj {
				set comp_y [expr $jjj * $scale_2]
				# puts $jjj
				# puts $comp_y
				lappend comp_90 $comp_y
				
			}
		}
	} 
	puts "---------------------------------------------------------------------"
	# puts "comp_00 = $comp_00"
	puts "--------------Components were read successfully!---------------------"
	# puts "comp_90 = $comp_90"
	
	
	set result {}
	foreach x $comp_00 y $comp_90 {
		lappend result [expr {$x + $y}]
}
	# set rotdcomp [expr $comp_00 + $comp_90]
	# puts "comp_rotd = $result"
	set outFile [file join  $Output_path "rotd.txt"]
	set output [open $outFile w]
	puts $output $result
	close $output
	puts "------------Components were combined successfully!-------------------"
	puts "---------------------------------------------------------------------"
     # puts $rotdcomp 
}

