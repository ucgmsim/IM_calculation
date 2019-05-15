# Calculating Period                                                                                                                                                                          
set omega2 [eigen 1]
set Tperiod [expr 2*3.1415/sqrt($omega2)]
puts "T= $Tperiod"
