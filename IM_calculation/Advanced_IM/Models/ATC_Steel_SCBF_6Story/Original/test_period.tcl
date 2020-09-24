# Define the time step used to run the analysis using the central difference scheme
# model directories
set modelpath Models
set modeldir $modelpath/BRBF_12Story

set pi 3.14
set dt_factor 1
set periods_file [open $modeldir/Model_Info/periods.out]
while {[gets $periods_file line] >= 0} {set period $line}
close $periods_file
set dt_analysis [expr {$dt_factor*$period/$pi}]
puts $period
puts $dt_analysis