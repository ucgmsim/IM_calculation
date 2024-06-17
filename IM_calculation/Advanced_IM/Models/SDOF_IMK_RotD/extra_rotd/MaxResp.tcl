#
# Procedure for reading maximum response and save it to a txt file
#

proc MaxResp {Output_path_resp} { 

    # upvar $max_resp1 max_resp
    
	set input_resp [open $Output_path_resp r]	
	set lineList [split [read $input_resp] "\n"]
	# puts "lineList= $lineList"
	
	set n 0
	foreach line $lineList {
		incr n
		if {$n == 2} {
			set max_resp [lindex $line 1]
			puts "max_resp = $max_resp"			
			
		}
	
	}
	
	return $max_resp 
}