#########################################################################################################################
### DEFINE BRACE ELEMENTS

set Abrace1 17.6;   #HSS 12-1/2 X 0.5
set Abrace2 17.6;   #HSS 12-1/2 X 0.5
set Abrace3 15.8;   #HSS 11-1/4 X 0.5
set Abrace4 13.4;   #HSS 9-5/8 X 0.5
set Abrace5 10.2;   #HSS 9-5/8 X 0.375
set Abrace6 6.59;   #HSS 7-1/2 X 0.312

set Ibrace1 1750;   #W18x97
set Ibrace2 3100;   #W24x104
set Ibrace3 4020;   #W24x131
set Ibrace4 1330;   #W18x76
set Ibrace5 4580;   #W24x146
set Ibrace6 1330;   #W21x62

#set Pybrace1 [expr $Abrace1*58.8]
#set Pybrace2 [expr $Abrace2*58.8]
#set Pybrace3 [expr $Abrace3*58.8]
#set Pybrace4 [expr $Abrace4*58.8]
#set Pybrace5 [expr $Abrace5*58.8]
#set Pybrace6 [expr $Abrace6*58.8]

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
#element zeroLength 	    111222   1001   1002  -mat 31 2 -dir 1 2 -orient 1 1 0 -1 1 0  
element elasticBeamColumn   1122300   1002   1003   $Astiff1  $Es  $Ihinge  $Corotational
#element nonlinearBeamColumn 11223   1002   1003   $np_brace  $sec_used1a   $Corotational

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
#element zeroLength 	    331222   1007   1008  -mat 31 2 -dir 1 2 -orient -1 1 0 1 1 0  
element elasticBeamColumn   3122300   1008   1009   $Astiff1  $Es  $Ihinge  $Corotational
#element nonlinearBeamColumn 31223   1008   1009   $np_brace  $sec_used1b   $Corotational

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
#element zeroLength 	    113222   2001   2002  -mat 32 2 -dir 1 2 -orient 1 -1 0 -1 -1 0  
element elasticBeamColumn   1322300   2002   2003   $Astiff2  $Es  $Ihinge  $Corotational
#element nonlinearBeamColumn 13223   2002   2003   $np_brace  $sec_used2a   $Corotational

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
#element zeroLength 	    333222   2007   2008  -mat 32 2 -dir 1 2 -orient -1 -1 0 1 -1 0  
element elasticBeamColumn   3322300   2008   2009   $Astiff2  $Es  $Ihinge  $Corotational
#element nonlinearBeamColumn 33223   2008   2009   $np_brace  $sec_used2b   $Corotational

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
#element zeroLength 	    113242   3001   3002  -mat 33 2 -dir 1 2 -orient 1 1 0 -1 1 0  
element elasticBeamColumn   1324300   3002   3003   $Astiff3  $Es  $Ihinge  $Corotational
#element nonlinearBeamColumn 13243   3002   3003   $np_brace  $sec_used3a   $Corotational

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



element elasticBeamColumn   3324100   23     3007   $Astiff2  $Es  $Istiff2  $Corotational 
element zeroLength 	    33242   3007   3008  -mat 3 2 -dir 1 2 -orient -1 1 0 1 1 0 
#element zeroLength 	    333242   3007   3008  -mat 33 2 -dir 1 2 -orient -1 1 0 1 1 0   
element elasticBeamColumn   3324300   3008   3009   $Astiff3  $Es  $Ihinge  $Corotational
#element nonlinearBeamColumn 33243   3008   3009   $np_brace  $sec_used3b   $Corotational

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
element elasticBeamColumn   3324700   3012   44     $Astiff2  $Es  $Istiff2  $Corotational

#############################################################################################
######## Story 4

set Astiff4 [expr $Abrace4*$stiffMult]
set Istiff4 [expr $Ibrace4*$IstiffMult]

element elasticBeamColumn   1524100   15     4001   $Astiff4  $Es  $Istiff4  $Corotational
element zeroLength 	    15242   4001   4002  -mat 3 2 -dir 1 2 -orient 1 -1 0 -1 -1 0  
#element zeroLength 	    115242   4001   4002  -mat 34 2 -dir 1 2 -orient 1 -1 0 -1 -1 0  
element elasticBeamColumn   1524300   4002   4003   $Astiff4  $Es  $Ihinge  $Corotational
#element nonlinearBeamColumn 15243   4002   4003   $np_brace  $sec_used4a   $Corotational

element nonlinearBeamColumn 1524401   4003     400301   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 1524402   400301   400302   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 1524403   400302   400303   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 1524404   400303   400304   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 1524405   400304   400305   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 1524406   400305   4004     $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 1524501   4004     400401   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 1524502   400401   400402   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 1524503   400402   400403   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 1524504   400403   400404   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 1524505   400404   400405   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 1524506   400405   4005     $np_brace  $sec_used4   $Corotational

element elasticBeamColumn   1524600   4005   4006   $Astiff4  $Es  $Ihinge  $Corotational
element elasticBeamColumn   1524700   4006   44     $Astiff4  $Es  $Istiff4  $Corotational  

element elasticBeamColumn   3524100   25     4007   $Astiff4  $Es  $Istiff4  $Corotational 
element zeroLength 	    35242   4007   4008  -mat 3 2 -dir 1 2 -orient -1 -1 0 1 -1 0  
#element zeroLength 	    335242   4007   4008  -mat 34 2 -dir 1 2 -orient -1 -1 0 1 -1 0  
element elasticBeamColumn   3524300   4008   4009   $Astiff4  $Es  $Ihinge  $Corotational
#element nonlinearBeamColumn 35243   4008   4009   $np_brace  $sec_used4b   $Corotational

element nonlinearBeamColumn 3524401   4009     400901   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 3524402   400901   400902   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 3524403   400902   400903   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 3524404   400903   400904   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 3524405   400904   400905   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 3524406   400905   4010     $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 3524501   4010     401001   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 3524502   401001   401002   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 3524503   401002   401003   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 3524504   401003   401004   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 3524505   401004   401005   $np_brace  $sec_used4   $Corotational
element nonlinearBeamColumn 3524506   401005   4011     $np_brace  $sec_used4   $Corotational

element elasticBeamColumn   3524600   4011   4012   $Astiff4  $Es  $Ihinge  $Corotational
element elasticBeamColumn   3524700   4012   44     $Astiff4  $Es  $Istiff4  $Corotational

#############################################################################################
######## Story 5

set Astiff5 [expr $Abrace5*$stiffMult]
set Istiff5 [expr $Ibrace5*$IstiffMult]

element elasticBeamColumn   1526100   15     5001   $Astiff5  $Es  $Istiff5  $Corotational
element zeroLength 	    15262   5001   5002  -mat 3 2 -dir 1 2 -orient 1 1 0 -1 1 0  
#element zeroLength 	    115262   5001   5002  -mat 35 2 -dir 1 2 -orient 1 1 0 -1 1 0  
element elasticBeamColumn   1526300   5002   5003   $Astiff5  $Es  $Ihinge  $Corotational
#element nonlinearBeamColumn 15263   5002   5003   $np_brace  $sec_used5a   $Corotational

element nonlinearBeamColumn 1526401   5003     500301   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 1526402   500301   500302   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 1526403   500302   500303   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 1526404   500303   500304   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 1526405   500304   500305   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 1526406   500305   5004     $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 1526501   5004     500401   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 1526502   500401   500402   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 1526503   500402   500403   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 1526504   500403   500404   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 1526505   500404   500405   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 1526506   500405   5005     $np_brace  $sec_used5   $Corotational

element elasticBeamColumn   1526600   5005   5006   $Astiff5  $Es  $Ihinge  $Corotational 
element elasticBeamColumn   1526700   5006   46     $Astiff5  $Es  $Istiff5  $Corotational  



element elasticBeamColumn   3526100   25     5007   $Astiff5  $Es  $Istiff5  $Corotational 
element zeroLength 	    35262   5007   5008  -mat 3 2 -dir 1 2 -orient -1 1 0 1 1 0
#element zeroLength 	    335262   5007   5008  -mat 35 2 -dir 1 2 -orient -1 1 0 1 1 0    
element elasticBeamColumn   3526300   5008   5009   $Astiff5  $Es  $Ihinge  $Corotational
#element nonlinearBeamColumn 35263   5008   5009   $np_brace  $sec_used5b   $Corotational

element nonlinearBeamColumn 3526401   5009     500901   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 3526402   500901   500902   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 3526403   500902   500903   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 3526404   500903   500904   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 3526405   500904   500905   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 3526406   500905   5010     $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 3526501   5010     501001   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 3526502   501001   501002   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 3526503   501002   501003   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 3526504   501003   501004   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 3526505   501004   501005   $np_brace  $sec_used5   $Corotational
element nonlinearBeamColumn 3526506   501005   5011     $np_brace  $sec_used5   $Corotational

element elasticBeamColumn   3526600   5011   5012   $Astiff5  $Es  $Ihinge  $Corotational
element elasticBeamColumn   3526700   5012   46     $Astiff5  $Es  $Istiff5  $Corotational


#############################################################################################
######## Story 6

set Astiff6 [expr $Abrace6*$stiffMult]
set Istiff6 [expr $Ibrace6*$IstiffMult]

element elasticBeamColumn   1726100   17     6001   $Astiff6  $Es  $Istiff6  $Corotational

element zeroLength 	    17262   6001   6002  -mat 3 2 -dir 1 2 -orient 1 -1 0 -1 -1 0 
#element zeroLength 	    117262   6001   6002  -mat 36 2 -dir 1 2 -orient 1 -1 0 -1 -1 0 

element elasticBeamColumn   1726300   6002   6003   $Astiff6  $Es  $Ihinge  $Corotational
#element nonlinearBeamColumn 17263   6002   6003   $np_brace  $sec_used6a   $Corotational

element nonlinearBeamColumn 1726401   6003     600301   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 1726402   600301   600302   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 1726403   600302   600303   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 1726404   600303   600304   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 1726405   600304   600305   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 1726406   600305   6004     $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 1726501   6004     600401   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 1726502   600401   600402   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 1726503   600402   600403   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 1726504   600403   600404   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 1726505   600404   600405   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 1726506   600405   6005     $np_brace  $sec_used6   $Corotational

element elasticBeamColumn   1726600   6005   6006   $Astiff6  $Es  $Ihinge  $Corotational
element elasticBeamColumn   1726700   6006   46     $Astiff6  $Es  $Istiff6  $Corotational  



element elasticBeamColumn   3726100   27     6007   $Astiff6  $Es  $Istiff6  $Corotational 
element zeroLength 	    37262   6007   6008  -mat 3 2 -dir 1 2 -orient -1 -1 0 1 -1 0  
#element zeroLength 	    337262   6007   6008  -mat 36 2 -dir 1 2 -orient -1 -1 0 1 -1 0 
element elasticBeamColumn   3726300   6008   6009   $Astiff6  $Es  $Ihinge  $Corotational
#element nonlinearBeamColumn 37263   6008   6009   $np_brace  $sec_used6b   $Corotational

element nonlinearBeamColumn 3726401   6009     600901   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 3726402   600901   600902   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 3726403   600902   600903   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 3726404   600903   600904   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 3726405   600904   600905   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 3726406   600905   6010     $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 3726501   6010     601001   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 3726502   601001   601002   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 3726503   601002   601003   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 3726504   601003   601004   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 3726505   601004   601005   $np_brace  $sec_used6   $Corotational
element nonlinearBeamColumn 3726506   601005   6011     $np_brace  $sec_used6   $Corotational

element elasticBeamColumn   3726600   6011   6012   $Astiff6  $Es  $Ihinge  $Corotational	
element elasticBeamColumn   3726700   6012   46     $Astiff6  $Es  $Istiff6  $Corotational


























                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    