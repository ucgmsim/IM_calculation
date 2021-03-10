set napaka 1e-3
set dt 0.01
set napaka2 1e-3
set dt2 0.01

while {$ok==0 && $currentTime<$tFinal} {
	set ok [analyze 1 $dt]
	if {$ok!=0} {
	        puts " Newton failed .. Trying Newton Initial .." 
		test NormDispIncr $napaka2 100 0
		algorithm Newton -initial
		set ok [analyze 1 $dt2]
		test NormDispIncr $napaka 12 0
		algorithm Newton  
		if {$ok == 0} {puts " that worked .. "}	
	}
	if {$ok!=0} {
		        puts " Newton failed .. Trying NewtonLineSearch .." 
			test NormDispIncr $napaka2 100 0
			algorithm NewtonLineSearch
			set ok [analyze 1 $dt2]
			test NormDispIncr $napaka 12 0
			algorithm Newton 
			if {$ok == 0} {puts " that worked .. "}	
	}
	if {$ok!=0} {
		puts " Newton failed .. Trying ModifiedNewton with Initial Tangent .." 
		test NormDispIncr $napaka2 100 0
		algorithm ModifiedNewton -initial
		set ok [analyze 1 $dt2]
		test NormDispIncr $napaka 12 0
		algorithm Newton
		if {$ok == 0} {puts " that worked .. "}	
	}
	set currentTime [getTime]
}