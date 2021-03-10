#########################################################################################################################
### DEFINE BRACE ELEMENTS

set Abrace1 13.4;  #HSS 9-5/8x.5
set Abrace2 13.4;  #HSS 9-5/8x.5
set Abrace3 13.4;  #HSS 9-5/8x.5
set Abrace4 13.4;  #HSS 9-5/8x.5
set Abrace5 10.6;  #HSS 10x.375
set Abrace6 10.6;  #HSS 10x.375
set Abrace7 10.6;  #HSS 10x.375
set Abrace8 10.6;  #HSS 10x.375
set Abrace9 7.73;  #HSS 8-3/4x.312
set Abrace10 7.73; #HSS 8-3/4x.312
set Abrace11 5.79; #HSS 6-5/8x.312
set Abrace12 5.79; #HSS 6-5/8x.312


set Ibrace1 1750;   #W18x97
set Ibrace2 3100;   #W24x104
set Ibrace3 4020;   #W24x131
set Ibrace4 1330;   #W18x76
set Ibrace5 4580;   #W24x146
set Ibrace6 1330;   #W21x62
set Ibrace7 1330;   #W21x62
set Ibrace8 1330;   #W21x62
set Ibrace9 1330;   #W21x62
set Ibrace10 1330;   #W21x62
set Ibrace11 1330;   #W21x62
set Ibrace12 1330;   #W21x62



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
set stiffMult 10.00;     #Multiplier on brace properties to create stiff member
set IstiffMult 10.00;

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

#############################################################################################
######## Story 4

set Astiff4 [expr $Abrace4*$stiffMult]
set Istiff4 [expr $Ibrace4*$IstiffMult]

element elasticBeamColumn   1524100   15     4001   $Astiff4  $Es  $Istiff4  $Corotational
element zeroLength 	    15242   4001   4002  -mat 3 2 -dir 1 2 -orient 1 -1 0 -1 -1 0    
element elasticBeamColumn   1524300   4002   4003   $Astiff4  $Es  $Ihinge  $Corotational

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
element elasticBeamColumn   3524300   4008   4009   $Astiff4  $Es  $Ihinge  $Corotational

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

element elasticBeamColumn   3524600  4011   4012   $Astiff4  $Es  $Ihinge  $Corotational
element elasticBeamColumn   3524700   4012   44     $Astiff4  $Es  $Istiff4  $Corotational

#############################################################################################
######## Story 5

set Astiff5 [expr $Abrace5*$stiffMult]
set Istiff5 [expr $Ibrace5*$IstiffMult]

element elasticBeamColumn   1526100   15     5001   $Astiff5  $Es  $Istiff5  $Corotational
element zeroLength 	    15262   5001   5002  -mat 3 2 -dir 1 2 -orient 1 1 0 -1 1 0  
element elasticBeamColumn   1526300   5002   5003   $Astiff5  $Es  $Ihinge  $Corotational

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
element elasticBeamColumn   3526300   5008   5009   $Astiff5  $Es  $Ihinge  $Corotational

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
element elasticBeamColumn   1726300   6002   6003   $Astiff6  $Es  $Ihinge  $Corotational

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
element elasticBeamColumn   3726300   6008   6009   $Astiff6  $Es  $Ihinge  $Corotational

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



#############################################################################################

#############################################################################################
######## Story 7

set Astiff7 [expr $Abrace7*$stiffMult]
set Istiff7 [expr $Ibrace7*$IstiffMult]


element elasticBeamColumn   1728100   17     7001   $Astiff7  $Es  $Istiff7  $Corotational
element zeroLength 	    17282   7001   7002  -mat 3 2 -dir 1 2 -orient 1 1 0 -1 1 0  
element elasticBeamColumn   1728300   7002   7003   $Astiff7  $Es  $Ihinge  $Corotational

element nonlinearBeamColumn 1728401   7003     700301   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 1728402   700301   700302   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 1728403   700302   700303   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 1728404   700303   700304   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 1728405   700304   700305   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 1728406   700305   7004     $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 1728501   7004     700401   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 1728502   700401   700402   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 1728503   700402   700403   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 1728504   700403   700404   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 1728505   700404   700405   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 1728506   700405   7005     $np_brace  $sec_used7   $Corotational

element elasticBeamColumn   1728600   7005   7006   $Astiff7  $Es  $Ihinge  $Corotational 
element elasticBeamColumn   1728700   7006   48     $Astiff7  $Es  $Istiff7  $Corotational  



element elasticBeamColumn   3728100   27     7007   $Astiff7  $Es  $Istiff7  $Corotational 
element zeroLength 	    37282   7007   7008  -mat 3 2 -dir 1 2 -orient -1 1 0 1 1 0
element elasticBeamColumn   3728300   7008   7009   $Astiff7  $Es  $Ihinge  $Corotational

element nonlinearBeamColumn 3728401   7009     700901   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 3728402   700901   700902   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 3728403   700902   700903   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 3728404   700903   700904   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 3728405   700904   700905   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 3728406   700905   7010     $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 3728501   7010     701001   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 3728502   701001   701002   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 3728503   701002   701003   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 3728504   701003   701004   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 3728505   701004   701005   $np_brace  $sec_used7   $Corotational
element nonlinearBeamColumn 3728506   701005   7011     $np_brace  $sec_used7   $Corotational

element elasticBeamColumn   3728600   7011   7012   $Astiff7  $Es  $Ihinge  $Corotational
element elasticBeamColumn   3728700   7012   48     $Astiff7  $Es  $Istiff7  $Corotational


#############################################################################################
######## Story 8

set Astiff8 [expr $Abrace8*$stiffMult]
set Istiff8 [expr $Ibrace8*$IstiffMult]


element elasticBeamColumn   1928100   19     8001   $Astiff8  $Es  $Istiff8  $Corotational
element zeroLength 	    19282   8001   8002  -mat 3 2 -dir 1 2 -orient 1 -1 0 -1 -1 0 
element elasticBeamColumn   1928300   8002   8003   $Astiff8  $Es  $Ihinge  $Corotational

element nonlinearBeamColumn 1928401   8003     800301   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 1928402   800301   800302   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 1928403   800302   800303   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 1928404   800303   800304   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 1928405   800304   800305   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 1928406   800305   8004     $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 1928501   8004     800401   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 1928502   800401   800402   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 1928503   800402   800403   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 1928504   800403   800404   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 1928505   800404   800405   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 1928506   800405   8005     $np_brace  $sec_used8   $Corotational

element elasticBeamColumn   1928600   8005   8006   $Astiff8  $Es  $Ihinge  $Corotational
element elasticBeamColumn   1928700   8006   48     $Astiff8  $Es  $Istiff8  $Corotational  



element elasticBeamColumn   3928100   29     8007   $Astiff8  $Es  $Istiff8  $Corotational 
element zeroLength 	    39282   8007   8008  -mat 3 2 -dir 1 2 -orient -1 -1 0 1 -1 0  
element elasticBeamColumn   3928300   8008   8009   $Astiff8  $Es  $Ihinge  $Corotational

element nonlinearBeamColumn 3928401   8009     800901   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 3928402   800901   800902   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 3928403   800902   800903   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 3928404   800903   800904   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 3928405   800904   800905   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 3928406   800905   8010     $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 3928501   8010     801001   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 3928502   801001   801002   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 3928503   801002   801003   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 3928504   801003   801004   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 3928505   801004   801005   $np_brace  $sec_used8   $Corotational
element nonlinearBeamColumn 3928506   801005   8011     $np_brace  $sec_used8   $Corotational

element elasticBeamColumn   3928600   8011   8012   $Astiff8  $Es  $Ihinge  $Corotational	
element elasticBeamColumn   3928700   8012   48     $Astiff8  $Es  $Istiff8  $Corotational


#############################################################################################
#############################################################################################
######## Story 9

set Astiff9 [expr $Abrace9*$stiffMult]
set Istiff9 [expr $Ibrace9*$IstiffMult]


element elasticBeamColumn   19210100   19     9001   $Astiff9  $Es  $Istiff9  $Corotational
element zeroLength 	    192102   9001   9002  -mat 3 2 -dir 1 2 -orient 1 1 0 -1 1 0  
element elasticBeamColumn   19210300   9002   9003   $Astiff9  $Es  $Ihinge  $Corotational

element nonlinearBeamColumn 19210401   9003     900301   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 19210402   900301   900302   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 19210403   900302   900303   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 19210404   900303   900304   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 19210405   900304   900305   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 19210406   900305   9004     $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 19210501   9004     900401   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 19210502   900401   900402   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 19210503   900402   900403   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 19210504   900403   900404   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 19210505   900404   900405   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 19210506   900405   9005     $np_brace  $sec_used9   $Corotational

element elasticBeamColumn   19210600   9005   9006   $Astiff9  $Es  $Ihinge  $Corotational 
element elasticBeamColumn   19210700   9006   410     $Astiff9  $Es  $Istiff9  $Corotational  



element elasticBeamColumn   39210100   29     9007   $Astiff9  $Es  $Istiff9  $Corotational 
element zeroLength 	    392102   9007   9008  -mat 3 2 -dir 1 2 -orient -1 1 0 1 1 0
element elasticBeamColumn   39210300   9008   9009   $Astiff9  $Es  $Ihinge  $Corotational

element nonlinearBeamColumn 39210401   9009     900901   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 39210402   900901   900902   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 39210403   900902   900903   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 39210404   900903   900904   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 39210405   900904   900905   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 39210406   900905   9010     $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 39210501   9010     901001   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 39210502   901001   901002   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 39210503   901002   901003   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 39210504   901003   901004   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 39210505   901004   901005   $np_brace  $sec_used9   $Corotational
element nonlinearBeamColumn 39210506   901005   9011     $np_brace  $sec_used9   $Corotational

element elasticBeamColumn   39210600   9011   9012   $Astiff9  $Es  $Ihinge  $Corotational
element elasticBeamColumn   39210700   9012   410     $Astiff9  $Es  $Istiff9  $Corotational


#############################################################################################
######## Story 10

set Astiff10 [expr $Abrace10*$stiffMult]
set Istiff10 [expr $Ibrace10*$IstiffMult]


element elasticBeamColumn   111210100   111     10001   $Astiff10  $Es  $Istiff10  $Corotational
element zeroLength 	    1112102   10001   10002  -mat 3 2 -dir 1 2 -orient 1 -1 0 -1 -1 0 
element elasticBeamColumn   111210300  10002   10003   $Astiff10  $Es  $Ihinge  $Corotational

element nonlinearBeamColumn 111210401   10003     1000301   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 111210402   1000301   1000302   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 111210403   1000302   1000303   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 111210404   1000303   1000304   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 111210405   1000304   1000305   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 111210406   1000305   10004     $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 111210501   10004     1000401   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 111210502   1000401   1000402   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 111210503   1000402   1000403   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 111210504   1000403   1000404   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 111210505   1000404   1000405   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 111210506   1000405   10005     $np_brace  $sec_used10   $Corotational

element elasticBeamColumn   111210600   10005   10006   $Astiff10  $Es  $Ihinge  $Corotational
element elasticBeamColumn   111210700   10006   410     $Astiff10  $Es  $Istiff10  $Corotational  



element elasticBeamColumn   311210100   211     10007   $Astiff10  $Es  $Istiff10  $Corotational 
element zeroLength 	    3112102   10007   10008  -mat 3 2 -dir 1 2 -orient -1 -1 0 1 -1 0  
element elasticBeamColumn   311210300   10008   10009   $Astiff10  $Es  $Ihinge  $Corotational

element nonlinearBeamColumn 311210401   10009     1000901   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 311210402   1000901   1000902   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 311210403   1000902   1000903   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 311210404   1000903   1000904   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 311210405   1000904   1000905   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 311210406   1000905   10010     $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 311210501   10010     1001001   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 311210502   1001001   1001002   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 311210503   1001002   1001003   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 311210504   1001003   1001004   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 311210505   1001004   1001005   $np_brace  $sec_used10   $Corotational
element nonlinearBeamColumn 311210506   1001005   10011     $np_brace  $sec_used10   $Corotational

element elasticBeamColumn   311210600   10011   10012   $Astiff10  $Es  $Ihinge  $Corotational	
element elasticBeamColumn   311210700   10012   410     $Astiff10  $Es  $Istiff10  $Corotational


#############################################################################################

#############################################################################################
#############################################################################################
######## Story 11

set Astiff11 [expr $Abrace11*$stiffMult]
set Istiff11 [expr $Ibrace11*$IstiffMult]


element elasticBeamColumn   111212100   111     11001   $Astiff11  $Es  $Istiff11  $Corotational
element zeroLength 	    1112122   11001   11002  -mat 3 2 -dir 1 2 -orient 1 1 0 -1 1 0  
element elasticBeamColumn   111212300   11002   11003   $Astiff11  $Es  $Ihinge  $Corotational

element nonlinearBeamColumn 111212401   11003     1100301   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 111212402   1100301   1100302   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 111212403   1100302   1100303   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 111212404   1100303   1100304   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 111212405   1100304   1100305   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 111212406   1100305   11004     $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 111212501   11004     1100401   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 111212502   1100401   1100402   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 111212503   1100402   1100403   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 111212504   1100403   1100404   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 111212505   1100404   1100405   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 111212506   1100405   11005     $np_brace  $sec_used11   $Corotational

element elasticBeamColumn   111212600   11005   11006   $Astiff11  $Es  $Ihinge  $Corotational 
element elasticBeamColumn   111212700   11006   412     $Astiff11  $Es  $Istiff11  $Corotational  



element elasticBeamColumn   311212100   211     11007   $Astiff11  $Es  $Istiff11  $Corotational 
element zeroLength 	    3112122   11007   11008  -mat 3 2 -dir 1 2 -orient -1 1 0 1 1 0  
element elasticBeamColumn   311212300   11008   11009   $Astiff11  $Es  $Ihinge  $Corotational

element nonlinearBeamColumn 311212401   11009     1100901   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 311212402   1100901   1100902   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 311212403   1100902   1100903   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 311212404   1100903   1100904   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 311212405   1100904   1100905   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 311212406   1100905   11010     $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 311212501   11010     1101001   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 311212502   1101001   1101002   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 311212503   1101002   1101003   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 311212504   1101003   1101004   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 311212505   1101004   1101005   $np_brace  $sec_used11   $Corotational
element nonlinearBeamColumn 311212506   1101005   11011     $np_brace  $sec_used11   $Corotational

element elasticBeamColumn   311212600   11011   11012   $Astiff11  $Es  $Ihinge  $Corotational
element elasticBeamColumn   311212700   11012   412     $Astiff11  $Es  $Istiff11  $Corotational


#############################################################################################
######## Story 12

set Astiff12 [expr $Abrace12*$stiffMult]
set Istiff12 [expr $Ibrace12*$IstiffMult]


element elasticBeamColumn   113212100   113     12001   $Astiff12  $Es  $Istiff12  $Corotational
element zeroLength 	    1132122   12001   12002  -mat 3 2 -dir 1 2 -orient 1 -1 0 -1 -1 0 
element elasticBeamColumn   113212300   12002   12003   $Astiff12  $Es  $Ihinge  $Corotational

element nonlinearBeamColumn 113212401   12003     1200301   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 113212402   1200301   1200302   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 113212403   1200302   1200303   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 113212404   1200303   1200304   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 113212405   1200304   1200305   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 113212406   1200305   12004     $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 113212501   12004     1200401   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 113212502   1200401   1200402   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 113212503   1200402   1200403   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 113212504   1200403   1200404   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 113212505   1200404   1200405   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 113212506   1200405   12005     $np_brace  $sec_used12   $Corotational

element elasticBeamColumn   113212600   12005   12006   $Astiff12  $Es  $Ihinge  $Corotational
element elasticBeamColumn   113212700   12006   412     $Astiff12  $Es  $Istiff12  $Corotational  



element elasticBeamColumn   313212100   213     12007   $Astiff12  $Es  $Istiff12  $Corotational 
element zeroLength 	    3132122   12007   12008  -mat 3 2 -dir 1 2 -orient -1 -1 0 1 -1 0  
element elasticBeamColumn   313212300   12008   12009   $Astiff12  $Es  $Ihinge  $Corotational

element nonlinearBeamColumn 313212401   12009     1200901   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 313212402   1200901   1200902   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 313212403   1200902   1200903   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 313212404   1200903   1200904   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 313212405   1200904   1200905   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 313212406   1200905   12010     $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 313212501   12010     1201001   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 313212502   1201001   1201002   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 313212503   1201002   1201003   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 313212504   1201003   1201004   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 313212505   1201004   1201005   $np_brace  $sec_used12   $Corotational
element nonlinearBeamColumn 313212506   1201005   12011     $np_brace  $sec_used12   $Corotational

element elasticBeamColumn   313212600   12011   12012   $Astiff12  $Es  $Ihinge  $Corotational	
element elasticBeamColumn   313212700   12012   412     $Astiff12  $Es  $Istiff12  $Corotational


#############################################################################################







































