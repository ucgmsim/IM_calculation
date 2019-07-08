proc getMaxResp {arrName} {
	upvar $arrName theVec
	if {![info exists theVec]} {
		return 0
	}
	set n [array size theVec]
	set resp 0.
	foreach index [array names theVec] {
		set val [recorderValue $theVec($index) 2]
		# puts "val= $val"
		if {$val > $resp} {
			set resp $val
		}
	}
	return $resp
}