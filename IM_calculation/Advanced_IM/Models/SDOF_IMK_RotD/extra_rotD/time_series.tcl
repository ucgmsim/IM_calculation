   # puts $outfile_path 
   set filePathX [lindex $outfile_path 0]
   set filePathY [lindex $outfile_path 1]
   puts "filePathX = $filePathX"
   puts "filePathY =  $filePathY"
   set fac [expr $g*1]
   set dt $dt1
   
   set filePath "$Output_path/rotd/rotd.txt"
   
   source [file join [file dirname [info script]] RotDcomp.tcl]
   RotDcomp $Output_path/rotd $filePathX $filePathY $thetaRad
   puts $filePath    
      
   timeSeries Path 7 -dt $dt -filePath $filePath -factor $fac  -startTime [getTime]
   pattern UniformExcitation  4   1  -accel 7