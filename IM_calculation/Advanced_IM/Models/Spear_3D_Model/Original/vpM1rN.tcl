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
 #             eleLoad	-ele	100	101	102	-type	-beamUniform	0	0 0
 #             eleLoad	-ele	200	201	202	-type	-beamUniform	0	0 0
 #             eleLoad	-ele	300	301	302	-type	-beamUniform	0	0 0
 #             eleLoad	-ele	400	401	402	-type	-beamUniform	0	0 0
 #             eleLoad	-ele	500	501	502	-type	-beamUniform	0	0 0
 #             eleLoad	-ele	600	601	602	-type	-beamUniform	0	0 0
 #             eleLoad	-ele	700	701	702	-type	-beamUniform	0	0 0
 #             eleLoad	-ele	800	801	802	-type	-beamUniform	0	0 0
 #             eleLoad	-ele	900	901	902	-type	-beamUniform	0	0 0
 #             eleLoad	-ele	1000	1001	1002	-type	-beamUniform	0	0 0
 #             eleLoad	-ele	1100	1101	1102	-type	-beamUniform	0	0 0
 #             eleLoad	-ele	1200	1201	1202	-type	-beamUniform	0	0 0
 #             eleLoad	-ele	1300	1301	1302	-type	-beamUniform	0	0 0
 #             eleLoad	-ele	1400	1401	1402	-type	-beamUniform	0	0 0
}

   test NormDispIncr 1.0e-2  50 1
   #test EnergyIncr 1.0e-4 1000 1
   algorithm Newton -initial 
   system SparseGeneral -piv
   numberer RCM
   constraints Transformation #Lagrange
   integrator LoadControl 1e-4
   #integrator LoadControl 0.01
   analysis Static
   analyze 100
   setTime 0.0
   
   set X1S11   "Series -dt 0.01 -filePath 001S11.txt -factor 1"
   set Y1S11   "Series -dt 0.01 -filePath 002S11.txt -factor 1"
   set R1S11   "Series -dt 0.01 -filePath 003S11.txt"

   set X2S11   "Series -dt 0.01 -filePath 004S11.txt -factor 1"
   set Y2S11   "Series -dt 0.01 -filePath 005S11.txt -factor 1"
   set R2S11   "Series -dt 0.01 -filePath 006S11.txt"

   set X3S11   "Series -dt 0.01 -filePath 007S11.txt -factor 1"
   set Y3S11   "Series -dt 0.01 -filePath 008S11.txt -factor 1"
   set R3S11   "Series -dt 0.01 -filePath 009S11.txt"

       
 ############################################
 ## "displacement-controlled" analysis     ##
 ############################################

   pattern MultipleSupport  2  {

       #             Zap.
       #              st.
       groundMotion     1   Plain  -disp  $X1S11
       groundMotion     2   Plain  -disp  $Y1S11
       groundMotion     3   Plain  -disp  $R1S11
       
       groundMotion     4   Plain  -disp  $X2S11
       groundMotion     5   Plain  -disp  $Y2S11
       groundMotion     6   Plain  -disp  $R2S11
       
       groundMotion     7   Plain  -disp  $X3S11
       groundMotion     8   Plain  -disp  $Y3S11
       groundMotion     9   Plain  -disp  $R3S11

       #             Vozl.  Smer   gMot.
       imposedMotion    52     1       1    
       imposedMotion    52     2       2   
       imposedMotion    52     6       3   
       
       imposedMotion    53     1       4   
       imposedMotion    53     2       5   
       imposedMotion    53     6       6   
       
       imposedMotion    54     1       7   
       imposedMotion    54     2       8   
       imposedMotion    54     6       9   
   }


   source RF3Ddisplay.tcl

      test NormDispIncr 1.0e-5 100 2
      #test EnergyIncr 1.0e-6 100 1
      #algorithm Newton -initial 
      algorithm Newton 
      #algorithm NewtonLineSearch 
      #NewtonLineSearch
      system SparseGeneral -piv
      numberer RCM
      constraints Transformation
      #Lagrange 1e20 1e20
      #Penalty 0.5 0.5
      #Transformation
      

#   recorder Node -file Node52.out -time -node 52 -dof 1 2 6 disp
#   recorder Node -file Node49.out -time -node 49 -dof 1 2 6 disp
#   recorder Node -file Node53.out -time -node 53 -dof 1 2 6 disp
#   recorder Node -file Node54.out -time -node 54 -dof 1 2 6 disp
#   recorder Node -file Node18.out -time -node 18 -dof 1 2 6 disp
#   recorder Node -file Node31.out -time -node 31 -dof 1 2 6 disp
#   recorder Node -file Node44.out -time -node 44 -dof 1 2 6 disp

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
   
#      recorder Element -time -file B1001LF.out -ele  1001 localForce #section 1 2 
#      recorder Element -time -file B1001LD1.out -ele  1001 section 1 deformation 
#      recorder Element -time -file B1001LD2.out -ele  1001 section 2 deformation
 

#    recorder Element -time -file C6LF.out -ele 6 localForce #section 1 2 
#    recorder Element -time -file C6LD1.out -ele 6 section 1 deformation
#    recorder Element -time -file C6LD2.out -ele 6 section 2 deformation 
#    recorder Element -time -file C3LF.out -ele 3 localForce #section 1 2 
#    recorder Element -time -file C3LD1.out -ele 3 section 1 deformation
#    recorder Element -time -file C3LD2.out -ele 3 section 2 deformation 
    
#    recorder Element -time -file B5LF.out -ele 500 localForce #section 1 2  
#    recorder Element -time -file B5LD1.out -ele 500 section 1 deformation
#    recorder Element -time -file B5LD2.out -ele 500 section 2 deformation 
#    recorder Element -time -file B10LF.out -ele 1000 localForce #section 
#    recorder Element -time -file B10LD1.out -ele 1000 section 1 deformation
#    recorder Element -time -file B10LD2.out -ele 1000 section 2 deformation 

#    recorder EnvelopeElement -time -file CE6LF.out -ele 6 localForce #section 1 2 
#    recorder EnvelopeElement -time -file CE6LD1.out -ele 6 section 1 deformation 
#    recorder EnvelopeElement -time -file CE6LD2.out -ele 6 section 2 deformation 
#    recorder EnvelopeElement -time -file CE3LF.out -ele 3 localForce #section 1 2 
#    recorder EnvelopeElement -time -file CE3LD1.out -ele 3 section 1 deformation 
#    recorder EnvelopeElement -time -file CE3LD2.out -ele 3 section 2 deformation    
    
#    recorder EnvelopeElement -time -file BE5LF.out -ele 500 localForce #section 1 2 
#    recorder EnvelopeElement -time -file BE5LD1.out -ele 500 section 1 deformation
#    recorder EnvelopeElement -time -file BE5LD2.out -ele 500 section 2 deformation 
#    recorder EnvelopeElement -time -file BE10LF.out -ele 1000 localForce #section 1 2 
#    recorder EnvelopeElement -time -file BE10LD1.out -ele 1000 section 1 deformation 
#    recorder EnvelopeElement -time -file BE10LD2.out -ele 1000 section 2 deformation 



#   recorder Element -time -file eleB10secDdef.out -ele 1000 section 2 deformation
#   recorder Element -time -file ele6secGforce.out -ele 6 globalForce #section 1 deformation

   #                   gama  beta   alfaM   betaK  betaKcomm   betaKinit
   integrator Newmark   0.5  0.25   0.0    0       0           0
   # 0.0     0.0        0.0         0.0


   analysis Transient
   #analyze 2000  0.01
   set tFinal 20.0;
   set ok 0;
   set currentTime 0.0;

   source Danaliza1N.tcl

   remove pattern 2
   setTime 0.0

   set X1S12   "Series -dt 0.01 -filePath 001S12.txt -factor 1"
   set Y1S12   "Series -dt 0.01 -filePath 002S12.txt -factor 1"
   set R1S12   "Series -dt 0.01 -filePath 003S12.txt"

   set X2S12   "Series -dt 0.01 -filePath 004S12.txt -factor 1"
   set Y2S12   "Series -dt 0.01 -filePath 005S12.txt -factor 1"
   set R2S12   "Series -dt 0.01 -filePath 006S12.txt"

   set X3S12   "Series -dt 0.01 -filePath 007S12.txt -factor 1"
   set Y3S12   "Series -dt 0.01 -filePath 008S12.txt -factor 1"
   set R3S12   "Series -dt 0.01 -filePath 009S12.txt"

   pattern MultipleSupport  3  {

       #             Zap.
       #              st.
       groundMotion     1   Plain  -disp  $X1S12
       groundMotion     2   Plain  -disp  $Y1S12
       groundMotion     3   Plain  -disp  $R1S12
       
       groundMotion     4   Plain  -disp  $X2S12
       groundMotion     5   Plain  -disp  $Y2S12
       groundMotion     6   Plain  -disp  $R2S12
       
       groundMotion     7   Plain  -disp  $X3S12
       groundMotion     8   Plain  -disp  $Y3S12
       groundMotion     9   Plain  -disp  $R3S12

       #             Vozl.  Smer   gMot.
       imposedMotion    52     1       1    
       imposedMotion    52     2       2   
       imposedMotion    52     6       3   
       
       imposedMotion    53     1       4   
       imposedMotion    53     2       5   
       imposedMotion    53     6       6   
       
       imposedMotion    54     1       7   
       imposedMotion    54     2       8   
       imposedMotion    54     6       9   
   }

 #  recorder Node -file Node18S12.out -time -node 18 -dof 1 2 6 disp


   set ok 0;
   set currentTime 0.0;

   source Danaliza1N.tcl
