# Argument 1: Shape name
# Argument 2: RBS? true/false

source ../Steel_Shapes/steel_wshapes.tcl
source atc72_equations.tcl

set shape [dict get $wshape_props [lindex $argv 0]]
dict with shape {set hinge [HingeProperties $d $bf_2tf $h_tw $ry [expr {$ry*50.0}] 150.0 55.0 [lindex $argv 1]]}

puts "thetap: [format %.3f [dict get $hinge thetap]]"
puts "thetapc: [format %.3f [dict get $hinge thetapc]]"
puts "Lambda: [format %.3f [dict get $hinge Lambda]]"
