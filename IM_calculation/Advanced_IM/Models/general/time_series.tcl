#
# --Define Time Series--
#
set filePath "../GMotions3/$GM_name/transformed/$station_name.txt"
set fac $g
set dt $dt1
timeSeries Path 5 -dt $dt -filePath $filePath -factor $fac  -startTime [getTime]
#pattern UniformExcitation 4 1 -accel "Series -dt $dt -filePath $filePath -factor $fac  -startTime [getTime]"
pattern UniformExcitation 4 1 -accel 5	
