#########################################################################################################################
### DEFINE BRACE ELEMENTS

set Abrace1 13.4;   #HSS 9-5/8 X 0.5
set Abrace2 12.1;   #HSS 8-3/4 X 0.5
set Abrace3 7.73;   #HSS 8-3/4 X 0.312


set Ibrace1 1750;   #W18x97
set Ibrace2 3100;   #W24x104
set Ibrace3 4020;   #W24x131



set np_brace 3;					# number of integration points for each brace element

set Corotational 2
geomTransf Corotational $Corotational


#############################################################################################
######## Story 1
#Element numbers are as follows: XX-YY-Z: 
# XX: Start master node
# YY: End master node
# Z: Number running from 1-7

# 0-th element is a stiff element to reduce effective length of brace
# 1-st element is a stiff element to reduce effective length of brace
# 2-nd element is a hinge with low EI
# 7-th element is a hinge with low EI
# 8-th element is a stiff element to reduce effective length of brace

set IhingePerc 0.005;  #Percentage of Ibrace that Ihinge gets
set stiffMult 7.00;     #Multiplier on brace properties to create stiff member
set IstiffMult 7.00;

set Astiff1 [expr $Abrace1*$stiffMult]
set Istiff1 [expr $Ibrace1*$IstiffMult]
set Ihinge [expr $Ibrace1*$IhingePerc]


element elasticBeamColumn   1122100   11     1001   $Astiff1  $Es  $Istiff1  $Corotational 
element zeroLength 	    11222   1001   1002  -mat 3 2 -dir 1 2 -orient 1 1 0 -1 1 0  
element elasticBeamColumn   1122300   1002   1003   $Astiff1  $Es  $Ihinge  $Corotational

element nonlinearBeamColumn 1122401   1003     100301   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 1122402   100301   100302   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 1122403   100302   100303   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 1122404   100303   100304   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 1122405   100304   100305   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 1122406   100305   1004     $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 1122501   1004     100401   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 1122502   100401   100402   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 1122503   100402   100403   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 1122504   100403   100404   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 1122505   100404   100405   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 1122506   100405   1005     $np_brace  $sec_used1   $Corotational

element elasticBeamColumn   1122600   1005   1006   $Astiff1  $Es  $Ihinge  $Corotational
element elasticBeamColumn   1122700   1006   42     $Astiff1  $Es  $Istiff1  $Corotational  



element elasticBeamColumn   3122100   21     1007   $Astiff1  $Es  $Istiff1  $Corotational  
element zeroLength 	    31222   1007   1008  -mat 3 2 -dir 1 2 -orient -1 1 0 1 1 0   
element elasticBeamColumn   3122300   1008   1009   $Astiff1  $Es  $Ihinge  $Corotational

element nonlinearBeamColumn 3122401   1009     100901   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 3122402   100901   100902   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 3122403   100902   100903   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 3122404   100903   100904   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 3122405   100904   100905   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 3122406   100905   1010     $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 3122501   1010     101001   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 3122502   101001   101002   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 3122503   101002   101003   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 3122504   101003   101004   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 3122505   101004   101005   $np_brace  $sec_used1   $Corotational
element nonlinearBeamColumn 3122506   101005   1011     $np_brace  $sec_used1   $Corotational

element elasticBeamColumn   3122600   1011   1012   $Astiff1  $Es  $Ihinge  $Corotational
element elasticBeamColumn   3122700   1012   42     $Astiff1  $Es  $Istiff1  $Corotational  


#############################################################################################
######## Story 2

set Astiff2 [expr $Abrace2*$stiffMult]
set Istiff2 [expr $Ibrace2*$IstiffMult]

element elasticBeamColumn   1322100   13     2001   $Astiff2  $Es  $Istiff2  $Corotational
element zeroLength 	    13222   2001   2002  -mat 3 2 -dir 1 2 -orient 1 -1 0 -1 -1 0   
element elasticBeamColumn   1322300   2002   2003   $Astiff2  $Es  $Ihinge  $Corotational

element nonlinearBeamColumn 1322401   2003     200301   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 1322402   200301   200302   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 1322403   200302   200303   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 1322404   200303   200304   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 1322405   200304   200305   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 1322406   200305   2004     $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 1322501   2004     200401   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 1322502   200401   200402   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 1322503   200402   200403   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 1322504   200403   200404   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 1322505   200404   200405   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 1322506   200405   2005     $np_brace  $sec_used2   $Corotational

element elasticBeamColumn   1322600   2005   2006   $Astiff2  $Es  $Ihinge  $Corotational
element elasticBeamColumn   1322700   2006   42     $Astiff2  $Es  $Istiff2  $Corotational  



element elasticBeamColumn   3322100   23     2007   $Astiff2  $Es  $Istiff2  $Corotational 
element zeroLength 	    33222   2007   2008  -mat 3 2 -dir 1 2 -orient -1 -1 0 1 -1 0  
element elasticBeamColumn   3322300   2008   2009   $Astiff2  $Es  $Ihinge  $Corotational

element nonlinearBeamColumn 3322401   2009     200901   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 3322402   200901   200902   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 3322403   200902   200903   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 3322404   200903   200904   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 3322405   200904   200905   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 3322406   200905   2010     $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 3322501   2010     201001   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 3322502   201001   201002   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 3322503   201002   201003   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 3322504   201003   201004   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 3322505   201004   201005   $np_brace  $sec_used2   $Corotational
element nonlinearBeamColumn 3322506   201005   2011     $np_brace  $sec_used2   $Corotational

element elasticBeamColumn   3322600   2011   2012   $Astiff2  $Es  $Ihinge  $Corotational
element elasticBeamColumn   3321700   2012   42     $Astiff2  $Es  $Istiff2  $Corotational

#############################################################################################
######## Story 3

set Astiff3 [expr $Abrace3*$stiffMult]
set Istiff3 [expr $Ibrace3*$IstiffMult]

element elasticBeamColumn   1324100   13     3001   $Astiff3  $Es  $Istiff3  $Corotational
element zeroLength 	    13242   3001   3002  -mat 3 2 -dir 1 2 -orient 1 1 0 -1 1 0  
element elasticBeamColumn   1324300   3002   3003   $Astiff3  $Es  $Ihinge  $Corotational

element nonlinearBeamColumn 1324401   3003     300301   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 1324402   300301   300302   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 1324403   300302   300303   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 1324404   300303   300304   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 1324405   300304   300305   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 1324406   300305   3004     $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 1324501   3004     300401   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 1324502   300401   300402   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 1324503   300402   300403   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 1324504   300403   300404   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 1324505   300404   300405   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 1324506   300405   3005     $np_brace  $sec_used3   $Corotational

element elasticBeamColumn   1324600   3005   3006   $Astiff3  $Es  $Ihinge  $Corotational
element elasticBeamColumn   1324700   3006   44     $Astiff3  $Es  $Istiff3  $Corotational  



element elasticBeamColumn   3324100   23     3007   $Astiff3  $Es  $Istiff3  $Corotational 
element zeroLength 	    33242   3007   3008  -mat 3 2 -dir 1 2 -orient -1 1 0 1 1 0  
element elasticBeamColumn   3324300   3008   3009   $Astiff3  $Es  $Ihinge  $Corotational

element nonlinearBeamColumn 3324401   3009     300901   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 3324402   300901   300902   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 3324403   300902   300903   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 3324404   300903   300904   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 3324405   300904   300905   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 3324406   300905   3010     $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 3324501   3010     301001   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 3324502   301001   301002   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 3324503   301002   301003   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 3324504   301003   301004   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 3324505   301004   301005   $np_brace  $sec_used3   $Corotational
element nonlinearBeamColumn 3324506   301005   3011     $np_brace  $sec_used3   $Corotational

element elasticBeamColumn   3324600   3011   3012   $Astiff3  $Es  $Ihinge  $Corotational
element elasticBeamColumn   3324700   3012   44     $Astiff3  $Es  $Istiff3  $Corotational


























                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    