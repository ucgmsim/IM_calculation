   # puts $outfile_path 
   set filePathX [lindex $outfile_path 0]
   set filePathY [lindex $outfile_path 1]
    # puts $filePathX
    # puts $filePathY
   set fac [expr $g*1]
   set dt $dt1
   
   timeSeries Path 5 -dt $dt -filePath $filePathX -factor $fac  -startTime [getTime]
   timeSeries Path 6 -dt $dt -filePath $filePathY -factor $fac  -startTime [getTime]
   #                        tag dir  accel series args
   pattern UniformExcitation  2   1  -accel 5
   pattern UniformExcitation  3   2  -accel 6
   
   