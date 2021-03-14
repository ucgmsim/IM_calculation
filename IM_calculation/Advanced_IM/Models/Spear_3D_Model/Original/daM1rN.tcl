#####################################
##  Vertical loading               ##
#####################################

# Reference lateral loads
pattern Plain 1 Constant {
#           eleID   uniform-                                           wy       wz    wx  (force/length)
       eleLoad	-ele	101	102	103	-type	-beamUniform	0	-6.78	0
       eleLoad	-ele	201	202	203	-type	-beamUniform	0	-9.07	0
       eleLoad	-ele	301	302	303	-type	-beamUniform	0	-10.22	0
       eleLoad	-ele	401	402	403	-type	-beamUniform	0	-15.23	0
       eleLoad	-ele	501	502	503	-type	-beamUniform	0	-6.78	0
       eleLoad	-ele	601	602	603	-type	-beamUniform	0	-9.38	0
       eleLoad	-ele	701	702	703	-type	-beamUniform	0	-17.88	0
       eleLoad	-ele	801	802	803	-type	-beamUniform	0	-11.01	0
       eleLoad	-ele	901	902	903	-type	-beamUniform	0	-15.12	0
       eleLoad	-ele	1001	1002	1003	-type	-beamUniform	0	-12.65	0
       eleLoad	-ele	1101	1102	1103	-type	-beamUniform	0	-8.32	0
       eleLoad	-ele	1201	1202	1203	-type	-beamUniform	0	-8.03	0
       eleLoad	-ele	1301	1302	1303	-type	-beamUniform	0	-9.38	0
       eleLoad	-ele	1401	1402	1403	-type	-beamUniform	0	-14.82	0
#              eleLoad	-ele	100	101	102	-type	-beamUniform	0	0 0
#              eleLoad	-ele	200	201	202	-type	-beamUniform	0	0 0
#              eleLoad	-ele	300	301	302	-type	-beamUniform	0	0 0
#              eleLoad	-ele	400	401	402	-type	-beamUniform	0	0 0
#              eleLoad	-ele	500	501	502	-type	-beamUniform	0	0 0
#              eleLoad	-ele	600	601	602	-type	-beamUniform	0	0 0
#              eleLoad	-ele	700	701	702	-type	-beamUniform	0	0 0
#              eleLoad	-ele	800	801	802	-type	-beamUniform	0	0 0
#              eleLoad	-ele	900	901	902	-type	-beamUniform	0	0 0
#              eleLoad	-ele	1000	1001	1002	-type	-beamUniform	0	0 0
#              eleLoad	-ele	1100	1101	1102	-type	-beamUniform	0	0 0
#              eleLoad	-ele	1200	1201	1202	-type	-beamUniform	0	0 0
#              eleLoad	-ele	1300	1301	1302	-type	-beamUniform	0	0 0
#              eleLoad	-ele	1400	1401	1402	-type	-beamUniform	0	0 0
}

   test NormDispIncr 1.0e-5  12 1
   #test EnergyIncr 1.0e-1 10 1
   algorithm Newton -initial 
   #algorithm Newton 
   system SparseGeneral -piv
   numberer RCM
   constraints Transformation #Lagrange
   integrator LoadControl 1e-4
   #integrator LoadControl 0.01
   analysis Static
   analyze 100
   setTime 0.0
           
#######################################
## NDA      S11-S12 (0.15g in 0.20g) ##
#######################################

 source RF3Ddisplay.tcl

 test NormDispIncr 1.0e-3 100 2
 #test EnergyIncr 1.0e-4 100 1
 #algorithm Newton -initial  
 algorithm Newton 
 #algorithm NewtonLineSearch 
 #NewtonLineSearch
 system SparseGeneral -piv
 #numberer Plain
 numberer RCM
 constraints Transformation
 #Lagrange 1e20 1e20
 #Penalty 0.5 0.5
 #Transformation

   #recorder Node -dT 0.01 -file VozliscaCM.out -time -node 52 53 54 -dof 1 2 6 disp
   recorder Node -file VozliscaCM.out -time -node 52 53 54 -dof 1 2 6 disp
   recorder Node -file VozliscaC1.out -time -node 24 37 50 -dof 1 2 6 disp
   recorder Node -file VozliscaC2.out -time -node 25 38 51 -dof 1 2 6 disp
   recorder Node -file VozliscaC3.out -time -node 18 31 44 -dof 1 2 6 disp
   recorder Node -file VozliscaC4.out -time -node 20 33 46 -dof 1 2 6 disp
   recorder Node -file VozliscaC5.out -time -node 23 36 49 -dof 1 2 6 disp
   recorder Node -file VozliscaC6.out -time -node 15 28 41 -dof 1 2 6 disp
   recorder Node -file VozliscaC7.out -time -node 17 30 43 -dof 1 2 6 disp
   recorder Node -file VozliscaC8.out -time -node 13 26 39 -dof 1 2 6 disp
   recorder Node -file VozliscaC9.out -time -node 21 34 47 -dof 1 2 6 disp
   
   
   recorder Element -time -file Stebri1GF.out -ele 1 2 3 4 5 6 7 8 9 globalForce #section 1 deformation
   recorder Element -time -file Stebri2GF.out -ele 10 11 12 13 14 15 16 17 18 globalForce #section 1 deformation
   recorder Element -time -file Stebri3GF.out -ele 19 20 21 22 23 24 25 26 27  globalForce #section 1 deformation
      
   recorder Element -time -file CH1F.out -ele 30101 30201 30301 30401 30501 30601 30701 30801 30901 31001 31101 31201 31301 31401 31501 31601 31701 31801 31901 32001 32101 32201 32301 32401 32501 32601 32701 section force 
   recorder Element -time -file CH1D.out -ele 30101 30201 30301 30401 30501 30601 30701 30801 30901 31001 31101 31201 31301 31401 31501 31601 31701 31801 31901 32001 32101 32201 32301 32401 32501 32601 32701 section deformation
   recorder Element -time -file CH2F.out -ele 30102 30202 30302 30402 30502 30602 30702 30802 30902 31002 31102 31202 31302 31402 31502 31602 31702 31802 31902 32002 32102 32202 32302 32402 32502 32602 32702 section force 
   recorder Element -time -file CH2D.out -ele 30102 30202 30302 30402 30502 30602 30702 30802 30902 31002 31102 31202 31302 31402 31502 31602 31702 31802 31902 32002 32102 32202 32302 32402 32502 32602 32702 section deformation
      
   recorder Element -time -file BH1F.out -ele 401011 402011 403011 404011 405011 406011 407011 408011 409011 410011 411011 412011 413011 414011 401021 402021 403021 404021 405021 406021 407021 408021 409021 410021 411021 412021 413021 414021 401031 402031 403031 404031 405031 406031 407031 408031 409031 410031 411031 412031 413031 414031 section force 
   recorder Element -time -file BH1D.out -ele 401011 402011 403011 404011 405011 406011 407011 408011 409011 410011 411011 412011 413011 414011 401021 402021 403021 404021 405021 406021 407021 408021 409021 410021 411021 412021 413021 414021 401031 402031 403031 404031 405031 406031 407031 408031 409031 410031 411031 412031 413031 414031 section deformation
   recorder Element -time -file BH2F.out -ele 401012 402012 403012 404012 405012 406012 407012 408012 409012 410012 411012 412012 413012 414012 401022 402022 403022 404022 405022 406022 407022 408022 409022 410022 411022 412022 413022 414022 401032 402032 403032 404032 405032 406032 407032 408032 409032 410032 411032 412032 413032 414032 section force 
   recorder Element -time -file BH2D.out -ele 401012 402012 403012 404012 405012 406012 407012 408012 409012 410012 411012 412012 413012 414012 401022 402022 403022 404022 405022 406022 407022 408022 409022 410022 411022 412022 413022 414022 401032 402032 403032 404032 405032 406032 407032 408032 409032 410032 411032 412032 413032 414032 section deformation

   #   recorder Element -time -file stebriLF.out -ele 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 localForce #section 1 2 
   #   recorder Element -time -file stebriLD1.out -ele 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 section 1 deformation
   #   recorder Element -time -file stebriLD2.out -ele 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 section 2 deformation
   #   recorder Element -time -file eleB5secLforce.out -ele 500 localForce #section 1 2 
   #   recorder Element -time -file gredeLF.out -ele  100 101 102 200 201 202 300 301 302 400 401 402 500 501 502 600 601 602 700 701 702 800 801 802 900 901 902 1000 1001 1002 1100 1101 1102 1200 1201 1202 1300 1301 1302 1400 1401 1402 localForce #section 1 2 
   #   recorder Element -time -file gredeLD1.out -ele  100 101 102 200 201 202 300 301 302 400 401 402 500 501 502 600 601 602 700 701 702 800 801 802 900 901 902 1000 1001 1002 1100 1101 1102 1200 1201 1202 1300 1301 1302 1400 1401 1402 section 1 deformation 
   #   recorder Element -time -file gredeLD2.out -ele  100 101 102 200 201 202 300 301 302 400 401 402 500 501 502 600 601 602 700 701 702 800 801 802 900 901 902 1000 1001 1002 1100 1101 1102 1200 1201 1202 1300 1301 1302 1400 1401 1402 section 2 deformation
  
  #  recorder Element -time -file C6LF.out -ele 6 localForce #section 1 2 
  #  recorder Element -time -file C6LD1.out -ele 6 section 1 deformation
  #  recorder Element -time -file C6LD2.out -ele 6 section 2 deformation #                   gama  beta   alfaM   betaK  betaKcomm   betaKinit
  #  recorder Element -time -file B1LF.out -ele 100 localForce #section 1 2  integrator Newmark   0.5  0.25   0.0    0       0           0
  #  recorder Element -time -file B1LD1.out -ele 100 section 1 deformation
  #  recorder Element -time -file B1LD2.out -ele 100 section 2 deformation analysis Transient
  #   #analyze 2000  0.01
  #  recorder EnvelopeElement -time -file CE6LF.out -ele 6 localForce #section 1 2  set tFinal 40.01;
  #  recorder EnvelopeElement -time -file CE6LD1.out -ele 6 section 1 deformation set ok 0;
  #  recorder EnvelopeElement -time -file CE6LD2.out -ele 6 section 2 deformation set currentTime 0.0;
  #  recorder EnvelopeElement -time -file BE1LF.out -ele 100 localForce #section 1 2 
  #  recorder EnvelopeElement -time -file BE1LD1.out -ele 100 section 1 deformation setTime 0.0
  #  recorder EnvelopeElement -time -file BE1LD2.out -ele 100 section 2 deformation 
    
      #                   gama  beta   alfaM   betaK  betaKcomm   betaKinit
      integrator Newmark   0.5  0.25   0.0    0       0           0
      # 0.0     0.0        0.0         0.0
      
         
      analysis Transient
      #analyze 2000  0.01
      set tFinal 40.01;
      set ok 0;
      set currentTime 0.0;
   
   setTime 0.0
   
   set accX "Path -filePath accXS11S12.txt -dt 0.01 -factor 1"
   set accY "Path -filePath accYS11S12.txt -dt 0.01 -factor 1"
   
   # Define the ground motion excitation using Tabas fault parallel and fault normal records
   #                         tag dir         accel series args
   pattern UniformExcitation  2   1  -accel    $accX
   pattern UniformExcitation  3   2  -accel    $accY

   #recorder Node -file Node18S12.out -time -node 18 -dof 1 2 6 disp

   set ok 0;
   set currentTime 0.0;

   source Danaliza1N.tcl
