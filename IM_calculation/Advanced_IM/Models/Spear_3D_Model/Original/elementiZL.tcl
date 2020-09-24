#######################################################
# COLUMNS     1-27 elastic elements; 30101-32702 hinges
#######################################################
# parameters of cross-section and material
set E 29471000;
set G [expr 0.4*$E];
set As1  [expr 0.25*0.25];                 # cross-section of column 25/25
set Is1  [expr 0.25*0.25*0.25*0.25/12];    # moment of inertia of column 25/25
set Ixs1 [expr 0.141*0.25*0.25*0.25*0.25]; # torsional mom. of inertia for column 25/25
set As2  [expr 0.25*0.75];                 # cross-section of column 25/75
set Iys2 [expr 0.25*0.25*0.25*0.75/12];    # moment of inertia in y direction of column 25/75 (weak direction)
set Izs2 [expr 0.25*0.75*0.75*0.75/12];    # moment of inertia in z direction of column 25/75 (strong direction)
set Ixs2 [expr 0.263*0.25*0.25*0.25*0.75]; # torsional mom. of inertia for column 25/25

# parameters for reducing of initial stiffness
set sf1 0.9;	
set sf2 0.8;	
set sf3 0.7;	
set sft 1.0;

# Transformation  tag vecxz  ----local z axis for columns in direction of global X axis
#geomTransf Linear 1   1 0 0
geomTransf PDelta 1   1 0 0

# columns 25/25     $eleTag $iNode $jNode $A $E $G $J $Iy $Iz $transfTag
# story 1
element elasticBeamColumn 1   1011  2024  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf1*$Is1]  [expr $sf1*$Is1]    1
element elasticBeamColumn 2   1012  2025  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf1*$Is1]  [expr $sf1*$Is1]    1
element elasticBeamColumn 3   1006  2018  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf1*$Is1]  [expr $sf1*$Is1]    1
element elasticBeamColumn 4   1007  2020  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf1*$Is1]  [expr $sf1*$Is1]    1
element elasticBeamColumn 5   1010  2023  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf1*$Is1]  [expr $sf1*$Is1]    1
element elasticBeamColumn 7   1005  2017  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf1*$Is1]  [expr $sf1*$Is1]    1
element elasticBeamColumn 8   1001  2013  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf1*$Is1]  [expr $sf1*$Is1]    1
element elasticBeamColumn 9   1008  2021  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf1*$Is1]  [expr $sf1*$Is1]    1
# story 2
element elasticBeamColumn 10  3024  4037  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf2*$Is1]  [expr $sf2*$Is1]      1
element elasticBeamColumn 11  3025  4038  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf2*$Is1]  [expr $sf2*$Is1]      1
element elasticBeamColumn 12  3018  4031  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf2*$Is1]  [expr $sf2*$Is1]      1
element elasticBeamColumn 13  3020  4033  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf2*$Is1]  [expr $sf2*$Is1]      1
element elasticBeamColumn 14  3023  4036  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf2*$Is1]  [expr $sf2*$Is1]      1
element elasticBeamColumn 16  3017  4030  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf2*$Is1]  [expr $sf2*$Is1]      1
element elasticBeamColumn 17  3013  4026  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf2*$Is1]  [expr $sf2*$Is1]      1
element elasticBeamColumn 18  3021  4034  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf2*$Is1]  [expr $sf2*$Is1]      1
# story 3
element elasticBeamColumn 19  5037  6050  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf3*$Is1]  [expr $sf3*$Is1]    1
element elasticBeamColumn 20  5038  6051  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf3*$Is1]  [expr $sf3*$Is1]    1
element elasticBeamColumn 21  5031  6044  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf3*$Is1]  [expr $sf3*$Is1]    1
element elasticBeamColumn 22  5033  6046  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf3*$Is1]  [expr $sf3*$Is1]    1
element elasticBeamColumn 23  5036  6049  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf3*$Is1]  [expr $sf3*$Is1]    1
element elasticBeamColumn 25  5030  6043  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf3*$Is1]  [expr $sf3*$Is1]    1
element elasticBeamColumn 26  5026  6039  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf3*$Is1]  [expr $sf3*$Is1]    1
element elasticBeamColumn 27  5034  6047  $As1  $E  $G  [expr $sft*$Ixs1]  [expr $sf3*$Is1]  [expr $sf3*$Is1]    1
# columns 25/75         $eleTag $iNode $jNode $A $E $G $Jz $Iy transfTag
element elasticBeamColumn  6  1003  2015  $As2  $E  $G  [expr $sft*$Ixs2]  [expr $sf1*$Iys2]  [expr $sf1*$Izs2]   1
element elasticBeamColumn 15  3015  4028  $As2  $E  $G  [expr $sft*$Ixs2]  [expr $sf2*$Iys2]  [expr $sf2*$Izs2]   1
element elasticBeamColumn 24  5028  6041  $As2  $E  $G  [expr $sft*$Ixs2]  [expr $sf3*$Iys2]  [expr $sf3*$Izs2]   1

# plastic hinges: story 1 at bottom (columns 25/25)
#                $eleTag $iNode $jNode -mat $matTag1 $matTag2 ... -dir $dir1 $dir2 ... <-orient $x1 $x2 $x3 $yp1 $yp2 $yp3>
element zeroLengthSection 30101 11   1011     1   -orient 0 0 1 0 -1 0  
element zeroLengthSection 30201 12   1012     2   -orient 0 0 1 0 -1 0
element zeroLengthSection 30301  6   1006     3   -orient 0 0 1 0 -1 0
element zeroLengthSection 30401  7   1007     4   -orient 0 0 1 0 -1 0
element zeroLengthSection 30501 10   1010     5   -orient 0 0 1 0 -1 0
element zeroLengthSection 30701  5   1005     7   -orient 0 0 1 0 -1 0
element zeroLengthSection 30801  1   1001     8   -orient 0 0 1 0 -1 0
element zeroLengthSection 30901  8   1008     9   -orient 0 0 1 0 -1 0
# plastic hinges: story 1 at top (columns 25/25)
element zeroLengthSection 30102  2024  24     51   -orient 0 0 1 0 -1 0 
element zeroLengthSection 30202  2025  25     52   -orient 0 0 1 0 -1 0 
element zeroLengthSection 30302  2018  18     53   -orient 0 0 1 0 -1 0 
element zeroLengthSection 30402  2020  20     54   -orient 0 0 1 0 -1 0 
element zeroLengthSection 30502  2023  23     55   -orient 0 0 1 0 -1 0
element zeroLengthSection 30702  2017  17     57   -orient 0 0 1 0 -1 0
element zeroLengthSection 30802  2013  13     58   -orient 0 0 1 0 -1 0
element zeroLengthSection 30902  2021  21     59   -orient 0 0 1 0 -1 0
# plastic hinges: story 2 at bottom (columns 25/25)
element zeroLengthSection 31001  24  3024     10   -orient 0 0 1 0 -1 0 
element zeroLengthSection 31101  25  3025     11   -orient 0 0 1 0 -1 0 
element zeroLengthSection 31201  18  3018     12   -orient 0 0 1 0 -1 0 
element zeroLengthSection 31301  20  3020     13   -orient 0 0 1 0 -1 0	
element zeroLengthSection 31401  23  3023     14   -orient 0 0 1 0 -1 0
element zeroLengthSection 31601  17  3017     16   -orient 0 0 1 0 -1 0
element zeroLengthSection 31701  13  3013     17   -orient 0 0 1 0 -1 0
element zeroLengthSection 31801  21  3021     18   -orient 0 0 1 0 -1 0
# plastic hinges: story 2 at top (columns 25/25)
element zeroLengthSection 31002  4037  37      510   -orient 0 0 1 0 -1 0 
element zeroLengthSection 31102  4038  38      511   -orient 0 0 1 0 -1 0 
element zeroLengthSection 31202  4031  31      512   -orient 0 0 1 0 -1 0 
element zeroLengthSection 31302  4033  33      513   -orient 0 0 1 0 -1 0 
element zeroLengthSection 31402  4036  36      514   -orient 0 0 1 0 -1 0
element zeroLengthSection 31602  4030  30      516   -orient 0 0 1 0 -1 0
element zeroLengthSection 31702  4026  26      517   -orient 0 0 1 0 -1 0
element zeroLengthSection 31802  4034  34      518   -orient 0 0 1 0 -1 0
# plastic hinges: story 3 at bottom (columns 25/25)
element zeroLengthSection 31901  37  5037     19   -orient 0 0 1 0 -1 0  
element zeroLengthSection 32001  38  5038     20   -orient 0 0 1 0 -1 0  
element zeroLengthSection 32101  31  5031     21   -orient 0 0 1 0 -1 0  
element zeroLengthSection 32201  33  5033     22   -orient 0 0 1 0 -1 0 	
element zeroLengthSection 32301  36  5036     23   -orient 0 0 1 0 -1 0	
element zeroLengthSection 32501  30  5030     25   -orient 0 0 1 0 -1 0	
element zeroLengthSection 32601  26  5026     26   -orient 0 0 1 0 -1 0
element zeroLengthSection 32701  34  5034     27   -orient 0 0 1 0 -1 0
# plastic hinges: story 3 at bottom (columns 25/25)
element zeroLengthSection 31902  6050  50      519   -orient 0 0 1 0 -1 0 
element zeroLengthSection 32002  6051  51      520   -orient 0 0 1 0 -1 0 
element zeroLengthSection 32102  6044  44      521   -orient 0 0 1 0 -1 0 
element zeroLengthSection 32202  6046  46      522   -orient 0 0 1 0 -1 0 
element zeroLengthSection 32302  6049  49      523   -orient 0 0 1 0 -1 0
element zeroLengthSection 32502  6043  43      525   -orient 0 0 1 0 -1 0
element zeroLengthSection 32602  6039  39      526   -orient 0 0 1 0 -1 0
element zeroLengthSection 32702  6047  47      527   -orient 0 0 1 0 -1 0
# plastic hinges for column C6 (strong column 25/75)				   
element zeroLengthSection 30601     3 1003       600   -orient 0 0 1 0 -1 0 
element zeroLengthSection 30602  2015   15      5600   -orient 0 0 1 0 -1 0 
element zeroLengthSection 31501    15 3015       150   -orient 0 0 1 0 -1 0 
element zeroLengthSection 31502  4028   28      5150   -orient 0 0 1 0 -1 0 
element zeroLengthSection 32401    28 5028       270   -orient 0 0 1 0 -1 0 
element zeroLengthSection 32402  6041   41      5270   -orient 0 0 1 0 -1 0  

# Transformation  tag vecxz  ---- beams: local z axis in direction of global Z axisza grede  lokalna z os v smeri globalne Z 
geomTransf Linear 2   0 0 1
geomTransf Linear 3   0 0 1

# beams story 1	       $eleTag $iNode $jNode $A           $E  $G  $J      $Iy                            $Iz                $transfTag
set sfI 1;
element elasticBeamColumn 101    10123 20124  [lindex $PG  0 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG  0 1]]  [lindex $PG  0 1]         2
element elasticBeamColumn 201    10224 20225  [lindex $PG  2 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG  2 1]]  [lindex $PG  2 1]         2
element elasticBeamColumn 301    10321 20322  [lindex $PG  4 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG  4 1]]  [lindex $PG  4 1]         2 
element elasticBeamColumn 401    10418 20419  [lindex $PG  6 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG  6 1]]  [lindex $PG  6 1]         2
element elasticBeamColumn 501    10513 20514  [lindex $PG  8 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG  8 1]]  [lindex $PG  8 1]         2
element elasticBeamColumn 601    10616 20617  [lindex $PG 10 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 10 1]]  [lindex $PG 10 1]         2
element elasticBeamColumn 701    10719 20725  [lindex $PG 12 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 12 1]]  [lindex $PG 12 1]         3
element elasticBeamColumn 801    10817 20820  [lindex $PG 14 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 14 1]]  [lindex $PG 14 1]         3
element elasticBeamColumn 901    10922 20924  [lindex $PG 16 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 16 1]]  [lindex $PG 16 1]         3
element elasticBeamColumn 1001   11016 21018  [lindex $PG 18 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 18 1]]  [lindex $PG 18 1]         3
element elasticBeamColumn 1101   11121 21123  [lindex $PG 20 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 20 1]]  [lindex $PG 20 1]         3
element elasticBeamColumn 1201   11213 21221  [lindex $PG 22 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 22 1]]  [lindex $PG 22 1]         3
element elasticBeamColumn 1301   11319 21320  [lindex $PG 24 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 24 1]]  [lindex $PG 24 1]         2
element elasticBeamColumn 1401   11418 21422  [lindex $PG 26 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 26 1]]  [lindex $PG 26 1]         3
# beams story 2
element elasticBeamColumn 102    10136 20137  [lindex $PG  0 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG  0 1]]  [lindex $PG  0 1]         2
element elasticBeamColumn 202    10237 20238  [lindex $PG  2 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG  2 1]]  [lindex $PG  2 1]         2
element elasticBeamColumn 302    10334 20335  [lindex $PG  4 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG  4 1]]  [lindex $PG  4 1]         2
element elasticBeamColumn 402    10431 20432  [lindex $PG  6 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG  6 1]]  [lindex $PG  6 1]         2
element elasticBeamColumn 502    10526 20527  [lindex $PG  8 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG  8 1]]  [lindex $PG  8 1]         2
element elasticBeamColumn 602    10629 20630  [lindex $PG 10 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 10 1]]  [lindex $PG 10 1]         2
element elasticBeamColumn 702    10732 20738  [lindex $PG 12 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 12 1]]  [lindex $PG 12 1]         3
element elasticBeamColumn 802    10830 20833  [lindex $PG 14 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 14 1]]  [lindex $PG 14 1]         3
element elasticBeamColumn 902    10935 20937  [lindex $PG 16 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 16 1]]  [lindex $PG 16 1]         3
element elasticBeamColumn 1002   11029 21031  [lindex $PG 18 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 18 1]]  [lindex $PG 18 1]         3
element elasticBeamColumn 1102   11134 21136  [lindex $PG 20 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 20 1]]  [lindex $PG 20 1]         3
element elasticBeamColumn 1202   11226 21234  [lindex $PG 22 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 22 1]]  [lindex $PG 22 1]         3
element elasticBeamColumn 1302   11332 21333  [lindex $PG 24 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 24 1]]  [lindex $PG 24 1]         2
element elasticBeamColumn 1402   11431 21435  [lindex $PG 26 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 26 1]]  [lindex $PG 26 1]         3
# beams story 3
element elasticBeamColumn 103    10149 20150  [lindex $PG  0 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG  0 1]]  [lindex $PG  0 1]         2
element elasticBeamColumn 203    10250 20251  [lindex $PG  2 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG  2 1]]  [lindex $PG  2 1]         2
element elasticBeamColumn 303    10347 20348  [lindex $PG  4 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG  4 1]]  [lindex $PG  4 1]         2
element elasticBeamColumn 403    10444 20445  [lindex $PG  6 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG  6 1]]  [lindex $PG  6 1]         2
element elasticBeamColumn 503    10539 20540  [lindex $PG  8 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG  8 1]]  [lindex $PG  8 1]         2
element elasticBeamColumn 603    10642 20643  [lindex $PG 10 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 10 1]]  [lindex $PG 10 1]         2
element elasticBeamColumn 703    10745 20751  [lindex $PG 12 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 12 1]]  [lindex $PG 12 1]         3
element elasticBeamColumn 803    10843 20846  [lindex $PG 14 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 14 1]]  [lindex $PG 14 1]         3
element elasticBeamColumn 903    10948 20950  [lindex $PG 16 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 16 1]]  [lindex $PG 16 1]         3
element elasticBeamColumn 1003   11042 21044  [lindex $PG 18 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 18 1]]  [lindex $PG 18 1]         3
element elasticBeamColumn 1103   11147 21149  [lindex $PG 20 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 20 1]]  [lindex $PG 20 1]         3
element elasticBeamColumn 1203   11239 21247  [lindex $PG 22 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 22 1]]  [lindex $PG 22 1]         3
element elasticBeamColumn 1303   11345 21346  [lindex $PG 24 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 24 1]]  [lindex $PG 24 1]         2
element elasticBeamColumn 1403   11444 21448  [lindex $PG 26 0]  $E  $G  2.1e-3  [expr $sfI*[lindex $PG 26 1]]  [lindex $PG 26 1]         3

# plastic hinges in beams of story 1 (at start)		       # x loc.  y loc.
element zeroLengthSection 401011  23   10123    1001   -orient -1  0 0   0 -1 0 
element zeroLengthSection 402011  24   10224    2001   -orient -1  0 0   0 -1 0 
element zeroLengthSection 403011  21   10321    3001   -orient -1  0 0   0 -1 0 
element zeroLengthSection 404011  18   10418    4001   -orient -1  0 0   0 -1 0 
element zeroLengthSection 405011  13   10513    5001   -orient -1  0 0   0 -1 0 
element zeroLengthSection 406011  16   10616    6001   -orient -1  0 0   0 -1 0 
element zeroLengthSection 407011  19   10719    7001   -orient  0 -1 0   1  0 0 
element zeroLengthSection 408011  17   10817    8001   -orient  0 -1 0   1  0 0 
element zeroLengthSection 409011  22   10922    9001   -orient  0 -1 0   1  0 0 
element zeroLengthSection 410011  16   11016    10001  -orient  0 -1 0   1  0 0 
element zeroLengthSection 411011  21   11121    11001  -orient  0 -1 0   1  0 0 
element zeroLengthSection 412011  13   11213    12001  -orient  0 -1 0   1  0 0 
element zeroLengthSection 413011  19   11319    13001  -orient -1  0 0   0 -1 0 
element zeroLengthSection 414011  18   11418    14001  -orient  0 -1 0   1  0 0 
# plastic hinges in beams of story 1 (at end)
element zeroLengthSection 401012  20124   24    1002   -orient -1  0 0   0 -1 0
element zeroLengthSection 402012  20225   25    2002   -orient -1  0 0   0 -1 0
element zeroLengthSection 403012  20322   22    3002   -orient -1  0 0   0 -1 0
element zeroLengthSection 404012  20419   19    4002   -orient -1  0 0   0 -1 0
element zeroLengthSection 405012  20514   14    5002   -orient -1  0 0   0 -1 0
element zeroLengthSection 406012  20617   17    6002   -orient -1  0 0   0 -1 0
element zeroLengthSection 407012  20725   25    7002   -orient  0 -1 0   1  0 0
element zeroLengthSection 408012  20820   20    8002   -orient  0 -1 0   1  0 0
element zeroLengthSection 409012  20924   24    9002   -orient  0 -1 0   1  0 0
element zeroLengthSection 410012  21018   18    10002  -orient  0 -1 0   1  0 0
element zeroLengthSection 411012  21123   23    11002  -orient  0 -1 0   1  0 0
element zeroLengthSection 412012  21221   21    12002  -orient  0 -1 0   1  0 0
element zeroLengthSection 413012  21320   20    13002  -orient -1  0 0   0 -1 0
element zeroLengthSection 414012  21422   22    14002  -orient  0 -1 0   1  0 0

# plastic hinges in beams of story 2 (at start)
element zeroLengthSection 401021  36  10136     1001   -orient -1  0 0   0 -1 0
element zeroLengthSection 402021  37  10237     2001   -orient -1  0 0   0 -1 0
element zeroLengthSection 403021  34  10334     3001   -orient -1  0 0   0 -1 0
element zeroLengthSection 404021  31  10431     4001   -orient -1  0 0   0 -1 0
element zeroLengthSection 405021  26  10526     5001   -orient -1  0 0   0 -1 0
element zeroLengthSection 406021  29  10629     6001   -orient -1  0 0   0 -1 0
element zeroLengthSection 407021  32  10732     7001   -orient  0 -1 0   1  0 0
element zeroLengthSection 408021  30  10830     8001   -orient  0 -1 0   1  0 0
element zeroLengthSection 409021  35  10935     9001   -orient  0 -1 0   1  0 0
element zeroLengthSection 410021  29  11029     10001  -orient  0 -1 0   1  0 0
element zeroLengthSection 411021  34  11134     11001  -orient  0 -1 0   1  0 0
element zeroLengthSection 412021  26  11226     12001  -orient  0 -1 0   1  0 0
element zeroLengthSection 413021  32  11332     13001  -orient -1  0 0   0 -1 0
element zeroLengthSection 414021  31  11431     14001  -orient  0 -1 0   1  0 0
# plastic hinges in beams of story 2 (at end)
element zeroLengthSection 401022  20137   37    1002   -orient -1  0 0   0 -1 0
element zeroLengthSection 402022  20238   38    2002   -orient -1  0 0   0 -1 0
element zeroLengthSection 403022  20335   35    3002   -orient -1  0 0   0 -1 0
element zeroLengthSection 404022  20432   32    4002   -orient -1  0 0   0 -1 0
element zeroLengthSection 405022  20527   27    5002   -orient -1  0 0   0 -1 0
element zeroLengthSection 406022  20630   30    6002   -orient -1  0 0   0 -1 0
element zeroLengthSection 407022  20738   38    7002   -orient  0 -1 0   1  0 0
element zeroLengthSection 408022  20833   33    8002   -orient  0 -1 0   1  0 0
element zeroLengthSection 409022  20937   37    9002   -orient  0 -1 0   1  0 0
element zeroLengthSection 410022  21031   31    10002  -orient  0 -1 0   1  0 0
element zeroLengthSection 411022  21136   36    11002  -orient  0 -1 0   1  0 0
element zeroLengthSection 412022  21234   34    12002  -orient  0 -1 0   1  0 0
element zeroLengthSection 413022  21333   33    13002  -orient -1  0 0   0 -1 0
element zeroLengthSection 414022  21435   35    14002  -orient  0 -1 0   1  0 0

# plastic hinges in beams of story 3 (at start)
element zeroLengthSection 401031  49  10149      1001   -orient -1  0 0   0 -1 0
element zeroLengthSection 402031  50  10250      2001   -orient -1  0 0   0 -1 0
element zeroLengthSection 403031  47  10347      3001   -orient -1  0 0   0 -1 0
element zeroLengthSection 404031  44  10444      4001   -orient -1  0 0   0 -1 0
element zeroLengthSection 405031  39  10539      5001   -orient -1  0 0   0 -1 0
element zeroLengthSection 406031  42  10642      6001   -orient -1  0 0   0 -1 0
element zeroLengthSection 407031  45  10745      7001   -orient  0 -1 0   1  0 0
element zeroLengthSection 408031  43  10843      8001   -orient  0 -1 0   1  0 0
element zeroLengthSection 409031  48  10948      9001   -orient  0 -1 0   1  0 0
element zeroLengthSection 410031  42  11042      10001  -orient  0 -1 0   1  0 0
element zeroLengthSection 411031  47  11147      11001  -orient  0 -1 0   1  0 0
element zeroLengthSection 412031  39  11239      12001  -orient  0 -1 0   1  0 0
element zeroLengthSection 413031  45  11345      13001  -orient -1  0 0   0 -1 0
element zeroLengthSection 414031  44  11444      14001  -orient  0 -1 0   1  0 0
# plastic hinges in beams of story 3 (at end)
element zeroLengthSection 401032  20150   50    1002   -orient -1  0 0   0 -1 0
element zeroLengthSection 402032  20251   51    2002   -orient -1  0 0   0 -1 0
element zeroLengthSection 403032  20348   48    3002   -orient -1  0 0   0 -1 0
element zeroLengthSection 404032  20445   45    4002   -orient -1  0 0   0 -1 0
element zeroLengthSection 405032  20540   40    5002   -orient -1  0 0   0 -1 0
element zeroLengthSection 406032  20643   43    6002   -orient -1  0 0   0 -1 0
element zeroLengthSection 407032  20751   51    7002   -orient  0 -1 0   1  0 0
element zeroLengthSection 408032  20846   46    8002   -orient  0 -1 0   1  0 0
element zeroLengthSection 409032  20950   50    9002   -orient  0 -1 0   1  0 0
element zeroLengthSection 410032  21044   44    10002  -orient  0 -1 0   1  0 0
element zeroLengthSection 411032  21149   49    11002  -orient  0 -1 0   1  0 0
element zeroLengthSection 412032  21247   47    12002  -orient  0 -1 0   1  0 0
element zeroLengthSection 413032  21346   46    13002  -orient -1  0 0   0 -1 0
element zeroLengthSection 414032  21448   48    14002  -orient  0 -1 0   1  0 0


# "rigid" elements connected to column 25/75
                         # tag  i j A   E    G	    J	Iy  Iz  transTag
element elasticBeamColumn 1500 14 15 1e6 1e10 0.4e10 1e6 1e6 1e6 2
element elasticBeamColumn 1600 15 16 1e6 1e10 0.4e10 1e6 1e6 1e6 2
element elasticBeamColumn 1501 27 28 1e6 1e10 0.4e10 1e6 1e6 1e6 2
element elasticBeamColumn 1601 28 29 1e6 1e10 0.4e10 1e6 1e6 1e6 2
element elasticBeamColumn 1502 40 41 1e6 1e10 0.4e10 1e6 1e6 1e6 2
element elasticBeamColumn 1602 41 42 1e6 1e10 0.4e10 1e6 1e6 1e6 2

# elements defining the diaphragm

set IyElastic 1e-8;
set IzElastic 1e4; 
set AElastic  1e6;
set Evelik    30e6;
set ItElastic 1e-8;
# beams story 1
 			   # tag    i j      A   E    G	  J  Iy        Iz         transTag 
element elasticBeamColumn 20011  52 13   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic 2
element elasticBeamColumn 20021  52 14   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic 2
element elasticBeamColumn 20031  52 15   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic 2
element elasticBeamColumn 20041  52 16   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic 2
element elasticBeamColumn 20051  52 17   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic 2
element elasticBeamColumn 20061  52 18   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic 2
element elasticBeamColumn 20071  52 19   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic 2
element elasticBeamColumn 20081  52 20   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic 2
element elasticBeamColumn 20091  52 21   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic 3
element elasticBeamColumn 20101  52 22   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic 3
element elasticBeamColumn 20111  52 23   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic 3
element elasticBeamColumn 20121  52 24   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic 3
element elasticBeamColumn 20131  52 25   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic 3
# beams story 2
element elasticBeamColumn 20012  53 26   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  2
element elasticBeamColumn 20022  53 27   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  2
element elasticBeamColumn 20032  53 28   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  2
element elasticBeamColumn 20042  53 29   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  2
element elasticBeamColumn 20052  53 30   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  2
element elasticBeamColumn 20062  53 31   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  2
element elasticBeamColumn 20072  53 32   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  3
element elasticBeamColumn 20082  53 33   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  3
element elasticBeamColumn 20092  53 34   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  3
element elasticBeamColumn 20102  53 35   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  3
element elasticBeamColumn 20112  53 36   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  3
element elasticBeamColumn 20122  53 37   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  3
element elasticBeamColumn 20132  53 38   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  2
# beams story 3
element elasticBeamColumn 20013  54 39   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  2
element elasticBeamColumn 20023  54 40   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  2
element elasticBeamColumn 20033  54 41   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  2
element elasticBeamColumn 20043  54 42   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  2
element elasticBeamColumn 20053  54 43   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  2
element elasticBeamColumn 20063  54 44   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  2
element elasticBeamColumn 20073  54 45   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  3
element elasticBeamColumn 20083  54 46   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  3
element elasticBeamColumn 20093  54 47   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  3
element elasticBeamColumn 20103  54 48   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  3
element elasticBeamColumn 20113  54 49   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  3
element elasticBeamColumn 20123  54 50   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  3
element elasticBeamColumn 20133  54 51   $AElastic $Evelik $Evelik $ItElastic $IyElastic $IzElastic  3











































































































































































































































































































































