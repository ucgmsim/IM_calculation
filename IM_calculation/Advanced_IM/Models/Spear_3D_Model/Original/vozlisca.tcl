# nodes for SPEAR building
model BasicBuilder -ndm 3 -ndf 6

# parameters of building geometry
set h1 2.75;              # story height  
set h2 3; 
set h3 3; 
set k1 [expr $h1];        
set k2 [expr $h1+$h2]; 
set k3 [expr $h1+$h2+$h3]; 

######################################
# COLUMN NODES
######################################
# C1
  node 11  0.0 -6.0  0.0
node 1011  0.0 -6.0  0.0
node 2024  0.0 -6.0  $k1
  node 24  0.0 -6.0  $k1
node 3024  0.0 -6.0  $k1
node 4037  0.0 -6.0  $k2
  node 37  0.0 -6.0  $k2 
node 5037  0.0 -6.0  $k2
node 6050  0.0 -6.0  $k3 
  node 50  0.0 -6.0  $k3  
# C2
  node 12 -5.0 -6.0  0.0
node 1012 -5.0 -6.0  0.0
node 2025 -5.0 -6.0  $k1 
  node 25 -5.0 -6.0  $k1
node 3025 -5.0 -6.0  $k1 
node 4038 -5.0 -6.0  $k2
  node 38 -5.0 -6.0  $k2
node 5038 -5.0 -6.0  $k2
node 6051 -5.0 -6.0  $k3
  node 51 -5.0 -6.0  $k3
# C3
  node 6   0.0  0.0  0.0
node 1006  0.0  0.0  0.0
node 2018  0.0  0.0  $k1
  node 18  0.0  0.0  $k1  
node 3018  0.0  0.0  $k1 
node 4031  0.0  0.0  $k2  
  node 31  0.0  0.0  $k2 
node 5031  0.0  0.0  $k2 
node 6044  0.0  0.0  $k3  
  node 44  0.0  0.0  $k3  
# C4
   node 7 -6.0  0.0  0.0
node 1007 -6.0  0.0  0.0  
node 2020 -6.0  0.0  $k1 
  node 20 -6.0  0.0  $k1
node 3020 -6.0  0.0  $k1 
node 4033 -6.0  0.0  $k2 
  node 33 -6.0  0.0  $k2
node 5033 -6.0  0.0  $k2
node 6046 -6.0  0.0  $k3 
  node 46 -6.0  0.0  $k3 
# C5
  node 10  3.0 -6.0  0.0 
node 1010  3.0 -6.0  0.0 
node 2023  3.0 -6.0  $k1
  node 23  3.0 -6.0  $k1
node 3023  3.0 -6.0  $k1
node 4036  3.0 -6.0  $k2 
  node 36  3.0 -6.0  $k2
node 5036  3.0 -6.0  $k2
node 6049  3.0 -6.0  $k3 
  node 49  3.0 -6.0  $k3 
# C6
   node 3  0.0  4.25 0.0
node 1003  0.0  4.25 0.0 
node 2015  0.0  4.25 $k1 
  node 15  0.0  4.25 $k1
node 3015  0.0  4.25 $k1 
node 4028  0.0  4.25 $k2 
  node 28  0.0  4.25 $k2
node 5028  0.0  4.25 $k2 
node 6041  0.0  4.25 $k3
  node 41  0.0  4.25 $k3 
# C7
   node 5 -6.0  4.0  0.0 
node 1005 -6.0  4.0  0.0 
node 2017 -6.0  4.0  $k1
  node 17 -6.0  4.0  $k1 
node 3017 -6.0  4.0  $k1
node 4030 -6.0  4.0  $k2
  node 30 -6.0  4.0  $k2
node 5030 -6.0  4.0  $k2 
node 6043 -6.0  4.0  $k3
  node 43 -6.0  4.0  $k3
# C8
   node 1  3.0  4.5  0.0
node 1001  3.0  4.5  0.0
node 2013  3.0  4.5  $k1
  node 13  3.0  4.5  $k1
node 3013  3.0  4.5  $k1 
node 4026  3.0  4.5  $k2
  node 26  3.0  4.5  $k2
node 5026  3.0  4.5  $k2
node 6039  3.0  4.5  $k3 
  node 39  3.0  4.5  $k3
# C9
   node 8  3.0 -0.5  0.0 
node 1008  3.0 -0.5  0.0  
node 2021  3.0 -0.5  $k1
  node 21  3.0 -0.5  $k1  
node 3021  3.0 -0.5  $k1 
node 4034  3.0 -0.5  $k2
  node 34  3.0 -0.5  $k2 
node 5034  3.0 -0.5  $k2 
node 6047  3.0 -0.5  $k3 
  node 47  3.0 -0.5  $k3

# additonal nodes in beams
node 14  0.0  4.5  $k1 # 1.etaza
node 16  0.0  4.0  $k1 
node 19 -5.0  0.0  $k1 
node 22  0.0 -0.5  $k1 
node 27  0.0  4.5  $k2 # 2.etaza 
node 29  0.0  4.0  $k2
node 32 -5.0  0.0  $k2
node 35  0.0 -0.5  $k2
node 40  0.0  4.5  $k3 # 3.etaza
node 42  0.0  4.0  $k3 
node 45 -5.0  0.0  $k3 
node 48  0.0 -0.5  $k3

# B1 [node i=1 or j=2 - story label - node number]  
node 10123  3.0 -6.0  $k1; node 20124   0.0 -6.0 $k1;
node 10136  3.0 -6.0  $k2; node 20137   0.0 -6.0 $k2;
node 10149  3.0 -6.0  $k3; node 20150   0.0 -6.0 $k3;
# B2
node 10224  0.0 -6.0  $k1; node 20225  -5.0 -6.0 $k1;
node 10237  0.0 -6.0  $k2; node 20238  -5.0 -6.0 $k2;
node 10250  0.0 -6.0  $k3; node 20251  -5.0 -6.0 $k3;
# B3
node 10321  3.0 -0.5  $k1; node 20322   0.0 -0.5 $k1; 
node 10334  3.0 -0.5  $k2; node 20335   0.0 -0.5 $k2; 
node 10347  3.0 -0.5  $k3; node 20348   0.0 -0.5 $k3; 
# B4
node 10418  0.0  0.0  $k1; node 20419  -5.0  0.0 $k1;  
node 10431  0.0  0.0  $k2; node 20432  -5.0  0.0 $k2;  
node 10444  0.0  0.0  $k3; node 20445  -5.0  0.0 $k3;  
# B5
node 10513  3.0  4.5  $k1; node 20514   0.0  4.5 $k1;
node 10526  3.0  4.5  $k2; node 20527   0.0  4.5 $k2;
node 10539  3.0  4.5  $k3; node 20540   0.0  4.5 $k3;
# B6
node 10616  0.0  4.0  $k1; node 20617  -6.0  4.0 $k1;
node 10629  0.0  4.0  $k2; node 20630  -6.0  4.0 $k2;
node 10642  0.0  4.0  $k3; node 20643  -6.0  4.0 $k3;
# B7
node 10719 -5.0  0.0  $k1; node 20725  -5.0 -6.0 $k1; 
node 10732 -5.0  0.0  $k2; node 20738  -5.0 -6.0 $k2; 
node 10745 -5.0  0.0  $k3; node 20751  -5.0 -6.0 $k3; 
# B8
node 10817 -6.0  4.0  $k1; node 20820  -6.0  0.0 $k1;
node 10830 -6.0  4.0  $k2; node 20833  -6.0  0.0 $k2;
node 10843 -6.0  4.0  $k3; node 20846  -6.0  0.0 $k3;
# B9
node 10922  0.0 -0.5  $k1; node 20924   0.0 -6.0 $k1;
node 10935  0.0 -0.5  $k2; node 20937   0.0 -6.0 $k2;
node 10948  0.0 -0.5  $k3; node 20950   0.0 -6.0 $k3;
# B10
node 11016  0.0  4.0  $k1; node 21018   0.0  0.0 $k1;
node 11029  0.0  4.0  $k2; node 21031   0.0  0.0 $k2;
node 11042  0.0  4.0  $k3; node 21044   0.0  0.0 $k3;
# B11
node 11121  3.0 -0.5  $k1; node 21123   3.0 -6.0 $k1;
node 11134  3.0 -0.5  $k2; node 21136   3.0 -6.0 $k2;
node 11147  3.0 -0.5  $k3; node 21149   3.0 -6.0 $k3;
# B12
node 11213  3.0  4.5  $k1; node 21221   3.0 -0.5 $k1; 
node 11226  3.0  4.5  $k2; node 21234   3.0 -0.5 $k2; 
node 11239  3.0  4.5  $k3; node 21247   3.0 -0.5 $k3; 
# B13
node 11319 -5.0  0.0  $k1; node 21320  -6.0  0.0 $k1; 
node 11332 -5.0  0.0  $k2; node 21333  -6.0  0.0 $k2;
node 11345 -5.0  0.0  $k3; node 21346  -6.0  0.0 $k3;
# B14
node 11418  0.0  0.0  $k1; node 21422   0.0 -0.5 $k1;  
node 11431  0.0  0.0  $k2; node 21435   0.0 -0.5 $k2;  
node 11444  0.0  0.0  $k3; node 21448   0.0 -0.5 $k3;

# Master nodes for rigid diaphragm (# center of mass)
set m1 67.264;
set m3 62.804 ;
set i1 1500.663;
set i3 1363.409;

#    tag  X             Y           Z 
node  52  [expr -1.58] [expr -0.85] $k1 -mass $m1 $m1 1e-6 1e-6 1e-6 $i1
node  53  [expr -1.58] [expr -0.85] $k2 -mass $m1 $m1 1e-6 1e-6 1e-6 $i1
node  54  [expr -1.65] [expr -0.94] $k3 -mass $m3 $m3 1e-6 1e-6 1e-6 $i3

# Set base constraints
#   tag DX DY DZ RX RY RZ
fix  1   1  1  1  1  1  1
fix  3   1  1  1  1  1  1
fix  5   1  1  1  1  1  1
fix  6   1  1  1  1  1  1
fix  7   1  1  1  1  1  1
fix  8   1  1  1  1  1  1
fix  10  1  1  1  1  1  1
fix  11  1  1  1  1  1  1
fix  12  1  1  1  1  1  1

########## Define Additional Information ##########

# Define the number of stories in the frame
set num_stories 3
set num_corners 4

# Define the control nodes used to compute story drifts
# CM nodes
set ctrl_nodes {
    6
    52
    53
    54
}

set ctrl_nodes_corner "ctrl_nodes_c1 ctrl_nodes_c2 ctrl_nodes_c3 ctrl_nodes_c4"

# corner nodes
set ctrl_nodes_c1 {
    1
    13
    26
    39
}

set ctrl_nodes_c2 {
    5
    17
    30
    43
}

set ctrl_nodes_c3 {
    10
    23
    36
    49
}
set ctrl_nodes_c4 {
    12
    25
    38
    51
}
 # puts [lindex [expr $[lindex $ctrl_nodes_corner 0]] 3]
    
    

