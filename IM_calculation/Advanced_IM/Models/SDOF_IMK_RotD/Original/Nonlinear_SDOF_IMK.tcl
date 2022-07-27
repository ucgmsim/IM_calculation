# --------------------------------------------------------------------------------------------------
# code to build SDOF system with peak-oriented IMK material 
# 

# SET UP ----------------------------------------------------------------------------
wipe;					# clear memory of all past model definitions
model BasicBuilder -ndm 1 -ndf 1;		# Define the model builder, ndm=#dimension, ndf=#dofs

# reading InPuts --------------------------------------------------------------------
puts [file join [file dirname [info script]]]
source [file join [file dirname [info script]] units_constants_metric.tcl]
source [file join [file dirname [info script]] params_backbone.tcl]
source [file join [file dirname [info script]] params_hysteresis.tcl]
source [file join [file dirname [info script]] Mat_paramfile.tcl]

# setting parameters & define Material ----------------------------------------------
set i 1
set imat_final 1;

set My [expr $My]
#puts My=$My
set My_neg [expr $My_neg]
#puts My_neg=$My_neg
set fi_y [expr $fi_y]
#puts fi_y=$fi_y
set ah [expr $ah]
#puts ah=$ah
set ac [expr $ac]
#puts ac=$ac
set mu_c [expr $mu_c]
#puts mu_c=$mu_c
set mu_f [expr $mu_f]
#puts mu_f=$mu_f
set rp [expr $rp]

set ePf1 $My
		puts ePf1=$ePf1
		set eNf1 $My_neg
		set ePd1 $fi_y 
		puts ePd1=$ePd1
		set eNd1 [expr $My_neg/($My/$fi_y)]
		puts eNf1=$eNf1
		puts eNd1=$eNd1


		set ePd2 [expr $fi_y*$mu_c] 

		set eNd2 [expr $eNd1*$mu_c]
		set ePf2 [expr $My*(($mu_c-1)*$ah+1)]
		puts ePf2=$ePf2 
		set eNf2 [expr $My_neg*(($mu_c_neg-1)*$ah+1)]
		puts ePd2=$ePd2
		puts eNf2=$eNf2
		puts eNd2=$eNd2
        set K0 [expr $ePf1/$ePd1]
		if {$rp == 0.000000001} {
			if {$ac < 0} {
			set ePf3 [expr +$ac*$K0*($mu_f*$fi_y-$ePd2)+$ePf2]
			set eNf3 [expr +$ac*$K0*($mu_f_neg*$eNd1-$eNd2)+$eNf2]
			} else {
			set ePf3 [expr $My*($ePf2/$My+($mu_f-$mu_c)*$ac)]
			set eNf3 [expr $My_neg*($eNf2/$My_neg+($mu_f_neg-$mu_c_neg)*$ac)]
			}

			set ePd3 [expr $fi_y*$mu_f]
			set eNd3 [expr $eNd1*$mu_f_neg]
			set ePf4 [expr 0.000000001]
			set ePd4 [expr $ePd3] 
			set eNf4 [expr 0.000000001]
			set eNd4 [expr $eNd3] 

		} else {
		set ePf3 [expr $My*$rp]
		set ePd3 [expr $fi_y*((($ePf2/$My-$rp)/$ac)+$mu_c)]
		
		set ePd3 [expr ($ePf2-$rp*$My+$mu_c*$fi_y*(-$ac*$K0))/(-$ac*$K0)]
		set ePf4 [expr $My*$rp]
		set ePd4 [expr $fi_y*$mu_f] 

		set eNf3 [expr $My_neg*$rp]
		set eNd3 [expr $eNd1*((($eNf2/$My_neg-$rp)/$ac)+$mu_c)]
		set eNd3 [expr ($eNf2-$rp*$My_neg+$mu_c_neg*$eNd1*(-$ac*$K0))/(-$ac*$K0)]
		set eNf4 [expr $My_neg*$rp]
		set eNd4 [expr $eNd1*$mu_f_neg] 
		} 
puts ePf3=$ePf3
puts ePd3=$ePd3 
puts eNf3=$eNf3
puts eNd3=$eNd3 
puts ePf4=$ePf4
puts ePd4=$ePd4 
puts eNf4=$eNf4
puts eNd4=$eNd4 
set ePd3_forIMK [expr (-$ePf2+($ac*$K0)*$ePd2)/($ac*$K0)]
set eNd3_forIMK [expr (-$eNf2+($ac*$K0)*$eNd2)/($ac*$K0)]


set ki [expr $ePf1/$ePd1]
set as_Plus [expr $ah]	
 #puts K0=$K0
 #puts as_Plus=$as_Plus
set as_Neg [expr $ah]	
 #puts as_Neg=$as_Neg
set My_Plus [expr $ePf1]
 #puts My_Plus=$My_Plus	
set My_Neg [expr $eNf1]
 #puts My_Neg=$My_Neg
# set Lamda_S 1500
# set Lamda_C 1500
# set Lamda_A 1500
# set Lamda_K 1500	
# set c_S 1	
# set c_C 1	
# set c_A 1		
# set c_K 1
set theta_p_Plus [expr $ePd2-$ePd1]	
 #puts theta_p_Plus=$theta_p_Plus
set theta_p_Neg [expr -($eNd2-$eNd1)]	
 #puts theta_p_Neg=$theta_p_Neg
set theta_pc_Plus [expr $ePd3_forIMK-$ePd2]
#puts theta_pc_Plus=$theta_pc_Plus
set theta_pc_Neg [expr -($eNd3_forIMK-$eNd2)]
#puts theta_pc_Neg=$theta_pc_Neg
set Res_Pos [expr $rp]	
#puts Res_Pos=$Res_Pos
set Res_Neg [expr $rp]	
set theta_u_Plus [expr $ePd4]
#puts theta_u_Plus=$theta_u_Plus
set theta_u_Neg [expr -$eNd4]
#puts theta_u_Neg=$theta_u_Neg
# set D_Plus 1
# set D_Neg 1
uniaxialMaterial ModIMKPeakOriented [expr $i*1] $K0 $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg

# System properties ----------------------------------------------
#-----------------------------------------  
#   if Tperiod <>0 set the mass to achieve the given period.
#   else set the period to match the given mass!
#-----------------------------------------
set kx $ki
if {$Tperiod!=0} {
   # set the mass now. It comes in "tons=1000kg"
   # Remember that T=2*pi*sqrt(m/k) <==> m = (T/(2pi))^2*k
   set mx [expr pow($Tperiod/(2*$pi),2)*$kx]
} else {
   set Tperiod 	[expr 2*$pi*sqrt($mx/$kx)]
}

set omega [expr pow($kx/$mx,0.5)]
puts "initial stiffness K=$kx"
puts "Mass              M=$mx"
puts "SDOF period       T=$Tperiod sec"
puts "cyclic frequency  W=$omega rad/sec"
puts "frequency         f=[expr 1.0/$Tperiod] Hz"

set outFile [open [file join [file dirname [info script]] Model_Info/periods.out] w]
puts $outFile $Tperiod
close $outFile

# geometry -------------------------------------------------------------------
# nodal coordinates:
node 1 0 
node 2 0  -mass $mx
element zeroLength 1 1 2 -mat $imat_final -dir 1
fix 1 1 

# Define damping -------------------------------------------------------------
set dampingtype "massproportional"
if {$dampingtype=="massproportional"} {
  # values taken from chopra, pg. 418.
  puts "Mass proportional damping [expr $ksi*100]%"
  puts "ksi=$ksi"
  set alphaM [expr $ksi*2.0*$omega]
  set betaKinit 0
} else {
  puts "Initial stiffness proportional damping [expr $ksi*100]%"
  set alphaM 0
  set betaKinit [expr $ksi*2.0/$omega]
}
# tangent stiffness proportional damping; +beatK*KCurrent
set betaK 0 
# last commited stiffness RAYLEIGH damping parameter; +betaKcomm*KlastCommitt
set betaKcomm 0

rayleigh $alphaM $betaK $betaKinit $betaKcomm

puts "rayleigh $alphaM $betaK $betaKinit $betaKcomm"
# ----------------------------------------------------------------------------
puts [pwd]
puts "Model Built"


 