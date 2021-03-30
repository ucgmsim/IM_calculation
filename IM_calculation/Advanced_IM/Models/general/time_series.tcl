#
# --Define Time Series--
#
set filePath $outfile
set fac [expr $g*1]
set dt $dt1
timeSeries Path 5 -dt $dt -filePath $filePath -factor $fac  -startTime [getTime]
pattern UniformExcitation 4 1 -accel 5	
