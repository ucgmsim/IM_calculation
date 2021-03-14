# moment-rotation envelopes and zero length section elements (HYSTERETIC MATERIAL)

# parameters of hysteretic material					
set pincX  0.0;   # columns
set pincY  0.0;    
set pincXg  0.0;  # beams
set pincYg  0.0;   
set pincXs  0.0;  # strong column at the top
set pincYs  0.0;   
set d1     0.0;   # columns
set d2g    0.0;   # beams
set d2s    0.0;   # strong column at the top
set betag  0.85;  # beams
set betas  0.75;  # columns
set betasm 0.85;  # strong column

# reads the moment rotation envelopes for top of columns in y direction into the list OvY
     set fdata [open "[file join [file dirname [info script]] stebriYZLz.txt]" r]
            while {[gets $fdata line] >= 0} {
                  lappend OvY $line
                  }
      close $fdata
# reads the moment rotation envelopes for bottom of columns in y direction into the list OvYs
     set fdata [open "[file join [file dirname [info script]] stebriYZL.txt]" r]
            while {[gets $fdata line] >= 0} {
                  lappend OvYs $line
                  }
      close $fdata
    
# reads the moment rotation envelopes for bottom of columns in z direction into the list OvZ
     set fdata [open "[file join [file dirname [info script]] stebriZZLz.txt]" r]
            while {[gets $fdata line] >= 0} {
                  lappend OvZ $line
                  }
      close $fdata  
# reads the moment rotation envelopes for top of columns in z direction into the list OvZs
     set fdata [open "[file join [file dirname [info script]] stebriZZL.txt]" r]
            while {[gets $fdata line] >= 0} {
                  lappend OvZs $line
                  }
      close $fdata        
# reads the moment rotation envelopes for beams in y direction into the list OvG
     #set fdata [open "gredeZLae.txt" r]
     set fdata [open "[file join [file dirname [info script]] gredeZLbe.txt]" r]
     #set fdata [open "gredeZLe.txt" r]
            while {[gets $fdata line] >= 0} {
                  lappend OvG $line
                  }
      close $fdata  
# reads A and I for beams
     #set fdata [open "PrereziGredZLae.txt" r]
     set fdata [open "[file join [file dirname [info script]] PrereziGredZLbe.txt]" r]
     #set fdata [open "PrereziGredZLe.txt" r]
            while {[gets $fdata line] >= 0} {
                  lappend PG $line
                  }
      close $fdata        

# elastic material for section
uniaxialMaterial Elastic 999 30e6; #axial and shear stiffness
uniaxialMaterial Elastic 998 1e-4; #torsional stiffness  

set ut 1.0;
					# Mp1      	    thp1		Mp2                 thp2           Mp3                      thp3                Mm1                 thp1                Mm2                 thp2                Mm3                  thp3
# columns 25/25 first story
uniaxialMaterial Hysteretic	1	[lindex $OvYs  0 0] [lindex $OvYs  0 1] [expr $ut*[lindex $OvYs  0 2]] [lindex $OvYs  0 3] [lindex $OvYs  0 4] [lindex $OvYs  0 5] [lindex $OvYs  0 6] [lindex $OvYs  0 7] [lindex $OvYs  0 8] [lindex $OvYs  0 9] [lindex $OvYs  0 10] [lindex $OvYs  0 11] $pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	2	[lindex $OvYs  1 0] [lindex $OvYs  1 1] [expr $ut*[lindex $OvYs  1 2]] [lindex $OvYs  1 3] [lindex $OvYs  1 4] [lindex $OvYs  1 5] [lindex $OvYs  1 6] [lindex $OvYs  1 7] [lindex $OvYs  1 8] [lindex $OvYs  1 9] [lindex $OvYs  1 10] [lindex $OvYs  1 11] $pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	3	[lindex $OvYs  2 0] [lindex $OvYs  2 1] [expr $ut*[lindex $OvYs  2 2]] [lindex $OvYs  2 3] [lindex $OvYs  2 4] [lindex $OvYs  2 5] [lindex $OvYs  2 6] [lindex $OvYs  2 7] [lindex $OvYs  2 8] [lindex $OvYs  2 9] [lindex $OvYs  2 10] [lindex $OvYs  2 11] $pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	4	[lindex $OvYs  3 0] [lindex $OvYs  3 1] [expr $ut*[lindex $OvYs  3 2]] [lindex $OvYs  3 3] [lindex $OvYs  3 4] [lindex $OvYs  3 5] [lindex $OvYs  3 6] [lindex $OvYs  3 7] [lindex $OvYs  3 8] [lindex $OvYs  3 9] [lindex $OvYs  3 10] [lindex $OvYs  3 11] $pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	5	[lindex $OvYs  4 0] [lindex $OvYs  4 1] [expr $ut*[lindex $OvYs  4 2]] [lindex $OvYs  4 3] [lindex $OvYs  4 4] [lindex $OvYs  4 5] [lindex $OvYs  4 6] [lindex $OvYs  4 7] [lindex $OvYs  4 8] [lindex $OvYs  4 9] [lindex $OvYs  4 10] [lindex $OvYs  4 11] $pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	7	[lindex $OvYs  6 0] [lindex $OvYs  6 1] [expr $ut*[lindex $OvYs  6 2]] [lindex $OvYs  6 3] [lindex $OvYs  6 4] [lindex $OvYs  6 5] [lindex $OvYs  6 6] [lindex $OvYs  6 7] [lindex $OvYs  6 8] [lindex $OvYs  6 9] [lindex $OvYs  6 10] [lindex $OvYs  6 11] $pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	8	[lindex $OvYs  7 0] [lindex $OvYs  7 1] [expr $ut*[lindex $OvYs  7 2]] [lindex $OvYs  7 3] [lindex $OvYs  7 4] [lindex $OvYs  7 5] [lindex $OvYs  7 6] [lindex $OvYs  7 7] [lindex $OvYs  7 8] [lindex $OvYs  7 9] [lindex $OvYs  7 10] [lindex $OvYs  7 11] $pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	9	[lindex $OvYs  8 0] [lindex $OvYs  8 1] [expr $ut*[lindex $OvYs  8 2]] [lindex $OvYs  8 3] [lindex $OvYs  8 4] [lindex $OvYs  8 5] [lindex $OvYs  8 6] [lindex $OvYs  8 7] [lindex $OvYs  8 8] [lindex $OvYs  8 9] [lindex $OvYs  8 10] [lindex $OvYs  8 11] $pincX	$pincY	$d1	$d2s	$betas

uniaxialMaterial Hysteretic	51	[lindex $OvY   0 0] [lindex $OvY   0 1] [expr $ut*[lindex $OvY   0 2]] [lindex $OvY   0 3] [lindex $OvY   0 4] [lindex $OvY   0 5] [lindex $OvY   0 6] [lindex $OvY   0 7] [lindex $OvY   0 8] [lindex $OvY   0 9] [lindex $OvY   0 10] [lindex $OvY   0 11] $pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	52	[lindex $OvY   1 0] [lindex $OvY   1 1] [expr $ut*[lindex $OvY   1 2]] [lindex $OvY   1 3] [lindex $OvY   1 4] [lindex $OvY   1 5] [lindex $OvY   1 6] [lindex $OvY   1 7] [lindex $OvY   1 8] [lindex $OvY   1 9] [lindex $OvY   1 10] [lindex $OvY   1 11] $pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	53	[lindex $OvY   2 0] [lindex $OvY   2 1] [expr $ut*[lindex $OvY   2 2]] [lindex $OvY   2 3] [lindex $OvY   2 4] [lindex $OvY   2 5] [lindex $OvY   2 6] [lindex $OvY   2 7] [lindex $OvY   2 8] [lindex $OvY   2 9] [lindex $OvY   2 10] [lindex $OvY   2 11] $pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	54	[lindex $OvY   3 0] [lindex $OvY   3 1] [expr $ut*[lindex $OvY   3 2]] [lindex $OvY   3 3] [lindex $OvY   3 4] [lindex $OvY   3 5] [lindex $OvY   3 6] [lindex $OvY   3 7] [lindex $OvY   3 8] [lindex $OvY   3 9] [lindex $OvY   3 10] [lindex $OvY   3 11] $pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	55	[lindex $OvY   4 0] [lindex $OvY   4 1] [expr $ut*[lindex $OvY   4 2]] [lindex $OvY   4 3] [lindex $OvY   4 4] [lindex $OvY   4 5] [lindex $OvY   4 6] [lindex $OvY   4 7] [lindex $OvY   4 8] [lindex $OvY   4 9] [lindex $OvY   4 10] [lindex $OvY   4 11] $pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	57	[lindex $OvY   6 0] [lindex $OvY   6 1] [expr $ut*[lindex $OvY   6 2]] [lindex $OvY   6 3] [lindex $OvY   6 4] [lindex $OvY   6 5] [lindex $OvY   6 6] [lindex $OvY   6 7] [lindex $OvY   6 8] [lindex $OvY   6 9] [lindex $OvY   6 10] [lindex $OvY   6 11] $pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	58	[lindex $OvY   7 0] [lindex $OvY   7 1] [expr $ut*[lindex $OvY   7 2]] [lindex $OvY   7 3] [lindex $OvY   7 4] [lindex $OvY   7 5] [lindex $OvY   7 6] [lindex $OvY   7 7] [lindex $OvY   7 8] [lindex $OvY   7 9] [lindex $OvY   7 10] [lindex $OvY   7 11] $pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	59	[lindex $OvY   8 0] [lindex $OvY   8 1] [expr $ut*[lindex $OvY   8 2]] [lindex $OvY   8 3] [lindex $OvY   8 4] [lindex $OvY   8 5] [lindex $OvY   8 6] [lindex $OvY   8 7] [lindex $OvY   8 8] [lindex $OvY   8 9] [lindex $OvY   8 10] [lindex $OvY   8 11] $pincX	$pincY	$d1	$d2s	$betas

#                      tag mat1 code1 mat2 code2 ... 
section Aggregator      1     1   My    1   Mz   998  T 999  P 999  Vy 999  Vz
section Aggregator      2     2   My    2   Mz   998  T	999  P 999  Vy 999  Vz
section Aggregator      3     3   My    3   Mz   998  T	999  P 999  Vy 999  Vz
section Aggregator      4     4   My    4   Mz   998  T	999  P 999  Vy 999  Vz
section Aggregator      5     5   My    5   Mz   998  T	999  P 999  Vy 999  Vz
section Aggregator      7     7   My    7   Mz   998  T	999  P 999  Vy 999  Vz
section Aggregator      8     8   My    8   Mz   998  T	999  P 999  Vy 999  Vz
section Aggregator      9     9   My    9   Mz   998  T	999  P 999  Vy 999  Vz

section Aggregator      51    51  My   51   Mz   998  T 999  P 999  Vy 999  Vz
section Aggregator      52    52  My   52   Mz   998  T 999  P 999  Vy 999  Vz
section Aggregator      53    53  My   53   Mz   998  T 999  P 999  Vy 999  Vz
section Aggregator      54    54  My   54   Mz   998  T 999  P 999  Vy 999  Vz
section Aggregator      55    55  My   55   Mz   998  T 999  P 999  Vy 999  Vz
section Aggregator      57    57  My   57   Mz   998  T 999  P 999  Vy 999  Vz
section Aggregator      58    58  My   58   Mz   998  T 999  P 999  Vy 999  Vz
section Aggregator      59    59  My   59   Mz   998  T 999  P 999  Vy 999  Vz

# columns 25/25 second story
uniaxialMaterial Hysteretic	10	[lindex $OvYs  9  0] [lindex $OvYs  9  1] [lindex $OvYs  9  2] [lindex $OvYs  9  3] [lindex $OvYs  9  4] [lindex $OvYs  9  5] [lindex $OvYs  9  6] [lindex $OvYs  9  7] [lindex $OvYs  9  8] [lindex $OvYs  9  9] [lindex $OvYs  9  10] [lindex $OvYs  9  11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	11	[lindex $OvYs  10 0] [lindex $OvYs  10 1] [lindex $OvYs  10 2] [lindex $OvYs  10 3] [lindex $OvYs  10 4] [lindex $OvYs  10 5] [lindex $OvYs  10 6] [lindex $OvYs  10 7] [lindex $OvYs  10 8] [lindex $OvYs  10 9] [lindex $OvYs  10 10] [lindex $OvYs  10 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	12	[lindex $OvYs  11 0] [lindex $OvYs  11 1] [lindex $OvYs  11 2] [lindex $OvYs  11 3] [lindex $OvYs  11 4] [lindex $OvYs  11 5] [lindex $OvYs  11 6] [lindex $OvYs  11 7] [lindex $OvYs  11 8] [lindex $OvYs  11 9] [lindex $OvYs  11 10] [lindex $OvYs  11 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	13	[lindex $OvYs  12 0] [lindex $OvYs  12 1] [lindex $OvYs  12 2] [lindex $OvYs  12 3] [lindex $OvYs  12 4] [lindex $OvYs  12 5] [lindex $OvYs  12 6] [lindex $OvYs  12 7] [lindex $OvYs  12 8] [lindex $OvYs  12 9] [lindex $OvYs  12 10] [lindex $OvYs  12 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	14	[lindex $OvYs  13 0] [lindex $OvYs  13 1] [lindex $OvYs  13 2] [lindex $OvYs  13 3] [lindex $OvYs  13 4] [lindex $OvYs  13 5] [lindex $OvYs  13 6] [lindex $OvYs  13 7] [lindex $OvYs  13 8] [lindex $OvYs  13 9] [lindex $OvYs  13 10] [lindex $OvYs  13 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	16	[lindex $OvYs  15 0] [lindex $OvYs  15 1] [lindex $OvYs  15 2] [lindex $OvYs  15 3] [lindex $OvYs  15 4] [lindex $OvYs  15 5] [lindex $OvYs  15 6] [lindex $OvYs  15 7] [lindex $OvYs  15 8] [lindex $OvYs  15 9] [lindex $OvYs  15 10] [lindex $OvYs  15 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	17	[lindex $OvYs  16 0] [lindex $OvYs  16 1] [lindex $OvYs  16 2] [lindex $OvYs  16 3] [lindex $OvYs  16 4] [lindex $OvYs  16 5] [lindex $OvYs  16 6] [lindex $OvYs  16 7] [lindex $OvYs  16 8] [lindex $OvYs  16 9] [lindex $OvYs  16 10] [lindex $OvYs  16 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	18	[lindex $OvYs  17 0] [lindex $OvYs  17 1] [lindex $OvYs  17 2] [lindex $OvYs  17 3] [lindex $OvYs  17 4] [lindex $OvYs  17 5] [lindex $OvYs  17 6] [lindex $OvYs  17 7] [lindex $OvYs  17 8] [lindex $OvYs  17 9] [lindex $OvYs  17 10] [lindex $OvYs  17 11]	$pincX	$pincY	$d1	$d2s	$betas

uniaxialMaterial Hysteretic	510	[lindex $OvY   9  0] [lindex $OvY   9  1] [lindex $OvY   9  2] [lindex $OvY   9  3] [lindex $OvY   9  4] [lindex $OvY   9  5] [lindex $OvY   9  6] [lindex $OvY   9  7] [lindex $OvY   9  8] [lindex $OvY   9  9] [lindex $OvY   9  10] [lindex $OvY   9  11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	511	[lindex $OvY   10 0] [lindex $OvY   10 1] [lindex $OvY   10 2] [lindex $OvY   10 3] [lindex $OvY   10 4] [lindex $OvY   10 5] [lindex $OvY   10 6] [lindex $OvY   10 7] [lindex $OvY   10 8] [lindex $OvY   10 9] [lindex $OvY   10 10] [lindex $OvY   10 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	512	[lindex $OvY   11 0] [lindex $OvY   11 1] [lindex $OvY   11 2] [lindex $OvY   11 3] [lindex $OvY   11 4] [lindex $OvY   11 5] [lindex $OvY   11 6] [lindex $OvY   11 7] [lindex $OvY   11 8] [lindex $OvY   11 9] [lindex $OvY   11 10] [lindex $OvY   11 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	513	[lindex $OvY   12 0] [lindex $OvY   12 1] [lindex $OvY   12 2] [lindex $OvY   12 3] [lindex $OvY   12 4] [lindex $OvY   12 5] [lindex $OvY   12 6] [lindex $OvY   12 7] [lindex $OvY   12 8] [lindex $OvY   12 9] [lindex $OvY   12 10] [lindex $OvY   12 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	514	[lindex $OvY   13 0] [lindex $OvY   13 1] [lindex $OvY   13 2] [lindex $OvY   13 3] [lindex $OvY   13 4] [lindex $OvY   13 5] [lindex $OvY   13 6] [lindex $OvY   13 7] [lindex $OvY   13 8] [lindex $OvY   13 9] [lindex $OvY   13 10] [lindex $OvY   13 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	516	[lindex $OvY   15 0] [lindex $OvY   15 1] [lindex $OvY   15 2] [lindex $OvY   15 3] [lindex $OvY   15 4] [lindex $OvY   15 5] [lindex $OvY   15 6] [lindex $OvY   15 7] [lindex $OvY   15 8] [lindex $OvY   15 9] [lindex $OvY   15 10] [lindex $OvY   15 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	517	[lindex $OvY   16 0] [lindex $OvY   16 1] [lindex $OvY   16 2] [lindex $OvY   16 3] [lindex $OvY   16 4] [lindex $OvY   16 5] [lindex $OvY   16 6] [lindex $OvY   16 7] [lindex $OvY   16 8] [lindex $OvY   16 9] [lindex $OvY   16 10] [lindex $OvY   16 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	518	[lindex $OvY   17 0] [lindex $OvY   17 1] [lindex $OvY   17 2] [lindex $OvY   17 3] [lindex $OvY   17 4] [lindex $OvY   17 5] [lindex $OvY   17 6] [lindex $OvY   17 7] [lindex $OvY   17 8] [lindex $OvY   17 9] [lindex $OvY   17 10] [lindex $OvY   17 11]	$pincX	$pincY	$d1	$d2s	$betas

#                      tag    mat1 code1 mat2 code2 ...
section Aggregator      10    10     My   10   Mz    998  T 999  P 999  Vy 999  Vz
section Aggregator      11    11     My   11   Mz    998  T 999  P 999  Vy 999  Vz
section Aggregator      12    12     My   12   Mz    998  T 999  P 999  Vy 999  Vz
section Aggregator      13    13     My   13   Mz    998  T 999  P 999  Vy 999  Vz
section Aggregator      14    14     My   14   Mz    998  T 999  P 999  Vy 999  Vz
section Aggregator      16    16     My   16   Mz    998  T 999  P 999  Vy 999  Vz
section Aggregator      17    17     My   17   Mz    998  T 999  P 999  Vy 999  Vz
section Aggregator      18    18     My   18   Mz    998  T 999  P 999  Vy 999  Vz

section Aggregator      510    510   My   510  Mz    998  T 999  P 999  Vy 999  Vz
section Aggregator      511    511   My   511  Mz    998  T 999  P 999  Vy 999  Vz
section Aggregator      512    512   My   512  Mz    998  T 999  P 999  Vy 999  Vz
section Aggregator      513    513   My   513  Mz    998  T 999  P 999  Vy 999  Vz
section Aggregator      514    514   My   514  Mz    998  T 999  P 999  Vy 999  Vz
section Aggregator      516    516   My   516  Mz    998  T 999  P 999  Vy 999  Vz
section Aggregator      517    517   My   517  Mz    998  T 999  P 999  Vy 999  Vz
section Aggregator      518    518   My   518  Mz    998  T 999  P 999  Vy 999  Vz

# columns 25/25 third story	
uniaxialMaterial Hysteretic	19	[lindex $OvYs  18 0] [lindex $OvYs  18 1] [lindex $OvYs  18 2] [lindex $OvYs  18 3] [lindex $OvYs  18 4] [lindex $OvYs  18 5] [lindex $OvYs  18 6] [lindex $OvYs  18 7] [lindex $OvYs  18 8] [lindex $OvYs  18 9] [lindex $OvYs  18 10] [lindex $OvYs  18 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	20	[lindex $OvYs  19 0] [lindex $OvYs  19 1] [lindex $OvYs  19 2] [lindex $OvYs  19 3] [lindex $OvYs  19 4] [lindex $OvYs  19 5] [lindex $OvYs  19 6] [lindex $OvYs  19 7] [lindex $OvYs  19 8] [lindex $OvYs  19 9] [lindex $OvYs  19 10] [lindex $OvYs  19 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	21	[lindex $OvYs  20 0] [lindex $OvYs  20 1] [lindex $OvYs  20 2] [lindex $OvYs  20 3] [lindex $OvYs  20 4] [lindex $OvYs  20 5] [lindex $OvYs  20 6] [lindex $OvYs  20 7] [lindex $OvYs  20 8] [lindex $OvYs  20 9] [lindex $OvYs  20 10] [lindex $OvYs  20 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	22	[lindex $OvYs  21 0] [lindex $OvYs  21 1] [lindex $OvYs  21 2] [lindex $OvYs  21 3] [lindex $OvYs  21 4] [lindex $OvYs  21 5] [lindex $OvYs  21 6] [lindex $OvYs  21 7] [lindex $OvYs  21 8] [lindex $OvYs  21 9] [lindex $OvYs  21 10] [lindex $OvYs  21 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	23	[lindex $OvYs  22 0] [lindex $OvYs  22 1] [lindex $OvYs  22 2] [lindex $OvYs  22 3] [lindex $OvYs  22 4] [lindex $OvYs  22 5] [lindex $OvYs  22 6] [lindex $OvYs  22 7] [lindex $OvYs  22 8] [lindex $OvYs  22 9] [lindex $OvYs  22 10] [lindex $OvYs  22 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	25	[lindex $OvYs  24 0] [lindex $OvYs  24 1] [lindex $OvYs  24 2] [lindex $OvYs  24 3] [lindex $OvYs  24 4] [lindex $OvYs  24 5] [lindex $OvYs  24 6] [lindex $OvYs  24 7] [lindex $OvYs  24 8] [lindex $OvYs  24 9] [lindex $OvYs  24 10] [lindex $OvYs  24 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	26	[lindex $OvYs  25 0] [lindex $OvYs  25 1] [lindex $OvYs  25 2] [lindex $OvYs  25 3] [lindex $OvYs  25 4] [lindex $OvYs  25 5] [lindex $OvYs  25 6] [lindex $OvYs  25 7] [lindex $OvYs  25 8] [lindex $OvYs  25 9] [lindex $OvYs  25 10] [lindex $OvYs  25 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	27	[lindex $OvYs  26 0] [lindex $OvYs  26 1] [lindex $OvYs  26 2] [lindex $OvYs  26 3] [lindex $OvYs  26 4] [lindex $OvYs  26 5] [lindex $OvYs  26 6] [lindex $OvYs  26 7] [lindex $OvYs  26 8] [lindex $OvYs  26 9] [lindex $OvYs  26 10] [lindex $OvYs  26 11]	$pincX	$pincY	$d1	$d2s	$betas

uniaxialMaterial Hysteretic	519	[lindex $OvY   18 0] [lindex $OvY   18 1] [lindex $OvY   18 2] [lindex $OvY   18 3] [lindex $OvY   18 4] [lindex $OvY   18 5] [lindex $OvY   18 6] [lindex $OvY   18 7] [lindex $OvY   18 8] [lindex $OvY   18 9] [lindex $OvY   18 10] [lindex $OvY   18 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	520	[lindex $OvY   19 0] [lindex $OvY   19 1] [lindex $OvY   19 2] [lindex $OvY   19 3] [lindex $OvY   19 4] [lindex $OvY   19 5] [lindex $OvY   19 6] [lindex $OvY   19 7] [lindex $OvY   19 8] [lindex $OvY   19 9] [lindex $OvY   19 10] [lindex $OvY   19 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	521	[lindex $OvY   20 0] [lindex $OvY   20 1] [lindex $OvY   20 2] [lindex $OvY   20 3] [lindex $OvY   20 4] [lindex $OvY   20 5] [lindex $OvY   20 6] [lindex $OvY   20 7] [lindex $OvY   20 8] [lindex $OvY   20 9] [lindex $OvY   20 10] [lindex $OvY   20 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	522	[lindex $OvY   21 0] [lindex $OvY   21 1] [lindex $OvY   21 2] [lindex $OvY   21 3] [lindex $OvY   21 4] [lindex $OvY   21 5] [lindex $OvY   21 6] [lindex $OvY   21 7] [lindex $OvY   21 8] [lindex $OvY   21 9] [lindex $OvY   21 10] [lindex $OvY   21 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	523	[lindex $OvY   22 0] [lindex $OvY   22 1] [lindex $OvY   22 2] [lindex $OvY   22 3] [lindex $OvY   22 4] [lindex $OvY   22 5] [lindex $OvY   22 6] [lindex $OvY   22 7] [lindex $OvY   22 8] [lindex $OvY   22 9] [lindex $OvY   22 10] [lindex $OvY   22 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	525	[lindex $OvY   24 0] [lindex $OvY   24 1] [lindex $OvY   24 2] [lindex $OvY   24 3] [lindex $OvY   24 4] [lindex $OvY   24 5] [lindex $OvY   24 6] [lindex $OvY   24 7] [lindex $OvY   24 8] [lindex $OvY   24 9] [lindex $OvY   24 10] [lindex $OvY   24 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	526	[lindex $OvY   25 0] [lindex $OvY   25 1] [lindex $OvY   25 2] [lindex $OvY   25 3] [lindex $OvY   25 4] [lindex $OvY   25 5] [lindex $OvY   25 6] [lindex $OvY   25 7] [lindex $OvY   25 8] [lindex $OvY   25 9] [lindex $OvY   25 10] [lindex $OvY   25 11]	$pincX	$pincY	$d1	$d2s	$betas
uniaxialMaterial Hysteretic	527	[lindex $OvY   26 0] [lindex $OvY   26 1] [lindex $OvY   26 2] [lindex $OvY   26 3] [lindex $OvY   26 4] [lindex $OvY   26 5] [lindex $OvY   26 6] [lindex $OvY   26 7] [lindex $OvY   26 8] [lindex $OvY   26 9] [lindex $OvY   26 10] [lindex $OvY   26 11]	$pincX	$pincY	$d1	$d2s	$betas

#                      tag    mat1 code1 mat2 code2 ...
section Aggregator      19    19     My  19   Mz  998  T 999  P 999  Vy 999  Vz
section Aggregator      20    20     My  20   Mz  998  T 999  P 999  Vy 999  Vz
section Aggregator      21    21     My  21   Mz  998  T 999  P 999  Vy 999  Vz
section Aggregator      22    22     My  22   Mz  998  T 999  P 999  Vy 999  Vz
section Aggregator      23    23     My  23   Mz  998  T 999  P 999  Vy 999  Vz
section Aggregator      25    25     My  25   Mz  998  T 999  P 999  Vy 999  Vz
section Aggregator      26    26     My  26   Mz  998  T 999  P 999  Vy 999  Vz
section Aggregator      27    27     My  27   Mz  998  T 999  P 999  Vy 999  Vz

section Aggregator      519    519   My  519  Mz  998  T  999  P 999  Vy 999  Vz
section Aggregator      520    520   My  520  Mz  998  T  999  P 999  Vy 999  Vz
section Aggregator      521    521   My  521  Mz  998  T  999  P 999  Vy 999  Vz
section Aggregator      522    522   My  522  Mz  998  T  999  P 999  Vy 999  Vz
section Aggregator      523    523   My  523  Mz  998  T  999  P 999  Vy 999  Vz
section Aggregator      525    525   My  525  Mz  998  T  999  P 999  Vy 999  Vz
section Aggregator      526    526   My  526  Mz  998  T  999  P 999  Vy 999  Vz
section Aggregator      527    527   My  527  Mz  998  T  999  P 999  Vy 999  Vz

# columns 25/75                 tag 601,151,271-1,2,3 story - weak dir., 602,152,272-1,2,3 story - strong direction
set sfthy 1.0;
set d2sMS 0;
set d1MS 0.0;


# decrasaing of initial stiffness in some hinges in strong columns (see post-test report)
set sfMS1  15.0;
set sfMS2  15.0;


					# Mp1      	    		  thp1		                     Mp2                  thp2                               Mp3                  thp3                                Mm1                             			    thp1                            			  	Mm2                  thp2                 			Mm3                   thp3
uniaxialMaterial Hysteretic	601	[lindex $OvYs  5  0] 		  [lindex $OvYs  5  1]               [lindex $OvYs  5  2] [lindex $OvYs  5  3]               [lindex $OvYs  5  4] [lindex $OvYs  5  5]               [lindex $OvYs  5  6] 	      			    [lindex $OvYs  5  7]              				[lindex $OvYs  5  8] [lindex $OvYs  5  9] 			[lindex $OvYs  5  10] [lindex $OvYs  5  11]  			$pincX	$pincY	$d1	$d2s	$betasm
#uniaxialMaterial Hysteretic	602	[lindex $OvZs  5  0]              [expr [lindex $OvZs  5  1]*$sfMS1] [lindex $OvZs  5  2] [expr [lindex $OvZs  5  3]*$sfMS1] [lindex $OvZs  5  4] [expr [lindex $OvZs  5  5]*$sfMS1] [lindex $OvZs  5  6] 			            [expr [lindex $OvZs  5  7]*$sfthy]				[lindex $OvZs  5  8] [lindex $OvZs  5  9] 			[lindex $OvZs  5  10] [lindex $OvZs  5  11]  			$pincX	$pincY	$d1MS	$d2sMS	$betasm
uniaxialMaterial Hysteretic	602	[lindex $OvZs  5  0]              [expr [lindex $OvZs  5  1]*$sfMS1] [lindex $OvZs  5  2] [expr [lindex $OvZs  5  3]*1.0000] [lindex $OvZs  5  4] [expr [lindex $OvZs  5  5]*1.0000] [lindex $OvZs  5  6] 			            [expr [lindex $OvZs  5  7]*$sfthy]				[lindex $OvZs  5  8] [lindex $OvZs  5  9] 			[lindex $OvZs  5  10] [lindex $OvZs  5  11]  			$pincXs	$pincYs	$d1MS	$d2sMS	$betasm

uniaxialMaterial Hysteretic	5601	[lindex $OvY   5  0] 		  [lindex $OvY   5  1]               [lindex $OvY   5  2] [lindex $OvY   5  3]               [lindex $OvY   5  4] [lindex $OvY   5  5]               [lindex $OvY   5  6] 			             [lindex $OvY   5  7] 		   			[lindex $OvY   5  8] [lindex $OvY   5  9] 			 [lindex $OvY   5  10] [lindex $OvY   5  11]  			$pincX	$pincY	$d1	$d2s	$betasm
uniaxialMaterial Hysteretic	5602	[lindex $OvZ   5  0] 		  [expr [lindex $OvZ   5  1]*$sfthy] [lindex $OvZ   5  2] [lindex $OvZ   5  3]               [lindex $OvZ   5  4] [lindex $OvZ   5  5]               [lindex $OvZ   5  6] 				     [expr [lindex $OvZ   5  7]*$sfthy] 			[lindex $OvZ   5  8] [lindex $OvZ   5  9] 			 [lindex $OvZ   5  10] [lindex $OvZ   5  11]  			$pincXs	$pincYs	$d1	$d2sMS	$betasm

uniaxialMaterial Hysteretic	151	[lindex $OvYs  14 0]     	  [lindex $OvYs  14 1]               [lindex $OvYs  14 2] [lindex $OvYs  14 3]               [lindex $OvYs  14 4] [lindex $OvYs  14 5]               [lindex $OvYs  14 6]				     [lindex $OvYs  14 7] 		    			[lindex $OvYs  14 8] [lindex $OvYs  14 9] 			 [lindex $OvYs  14 10] [lindex $OvYs  14 11] 			$pincX	$pincY	$d1	$d2s	$betasm
uniaxialMaterial Hysteretic	152	[lindex $OvZs  14 0] 		  [expr [lindex $OvZs  14 1]*$sfthy] [lindex $OvZs  14 2] [lindex $OvZs  14 3]               [lindex $OvZs  14 4] [lindex $OvZs  14 5]               [lindex $OvZs  14 6] 				     [expr [lindex $OvZs  14 7]*$sfthy] 			[lindex $OvZs  14 8] [lindex $OvZs  14 9] 			 [lindex $OvZs  14 10] [lindex $OvZs  14 11]  			$pincXs	$pincYs	$d1	$d2sMS	$betasm

uniaxialMaterial Hysteretic	5151	[lindex $OvY   14 0] 		  [lindex $OvY   14 1]               [lindex $OvY   14 2] [lindex $OvY   14 3]               [lindex $OvY   14 4] [lindex $OvY   14 5]               [lindex $OvY   14 6]				     [lindex $OvY   14 7] 		   			[lindex $OvY   14 8] [lindex $OvY   14 9] 			 [lindex $OvY   14 10] [lindex $OvY   14 11]  			$pincX	$pincY	$d1	$d2s	$betasm
#uniaxialMaterial Hysteretic	5152	[lindex $OvZ   14 0]              [expr [lindex $OvZ   14 1]*$sfthy] [lindex $OvZ   14 2] [lindex $OvZ   14 3]               [lindex $OvZ   14 4] [lindex $OvZ   14 5]               [lindex $OvZ   14 6] 			             [expr [lindex $OvZ   14 7]*$sfMS2] 			[lindex $OvZ   14 8] [expr [lindex $OvZ   14 9]*$sfMS2] 	 [lindex $OvZ   14 10] [expr [lindex $OvZ   14 11]*$sfMS2] 	$pincX	$pincY	$d1MS	$d2sMS	$betasm
uniaxialMaterial Hysteretic	5152	[lindex $OvZ   14 0]              [expr [lindex $OvZ   14 1]*$sfthy] [lindex $OvZ   14 2] [lindex $OvZ   14 3]               [lindex $OvZ   14 4] [lindex $OvZ   14 5]               [lindex $OvZ   14 6] 			             [expr [lindex $OvZ   14 7]*$sfMS2] 			[lindex $OvZ   14 8] [expr [lindex $OvZ   14 9]*1.0000] 	 [lindex $OvZ   14 10] [expr [lindex $OvZ   14 11]*1.0000] 	$pincXs	$pincYs	$d1MS	$d2sMS	$betasm

uniaxialMaterial Hysteretic	241	[lindex $OvYs  23 0] 		  [lindex $OvYs  23 1]               [lindex $OvYs  23 2] [lindex $OvYs  23 3]               [lindex $OvYs  23 4] [lindex $OvYs  23 5]               [lindex $OvYs  23 6] 				     [lindex $OvYs  23 7] 		    			[lindex $OvYs  23 8] [lindex $OvYs  23 9]			 [lindex $OvYs  23 10] [lindex $OvYs  23 11]  			$pincX	$pincY	$d1	$d2s	$betasm
uniaxialMaterial Hysteretic	242	[lindex $OvZs  23 0]		  [expr [lindex $OvZs  23 1]*$sfthy] [lindex $OvZs  23 2] [lindex $OvZs  23 3]               [lindex $OvZs  23 4] [lindex $OvZs  23 5]               [lindex $OvZs  23 6] 				     [expr [lindex $OvZs  23 7]*$sfthy] 			[lindex $OvZs  23 8] [lindex $OvZs  23 9] 			 [lindex $OvZs  23 10] [lindex $OvZs  23 11]  			$pincXs	$pincYs	$d1	$d2sMS	$betasm

uniaxialMaterial Hysteretic	5241	[lindex $OvY   23 0]		  [lindex $OvY   23 1]               [lindex $OvY   23 2] [lindex $OvY   23 3]               [lindex $OvY   23 4] [lindex $OvY   23 5]               [lindex $OvY   23 6]				     [lindex $OvY   23 7] 		   			[lindex $OvY   23 8] [lindex $OvY   23 9]			 [lindex $OvY   23 10] [lindex $OvY   23 11]  			$pincX	$pincY	$d1	$d2s	$betasm
uniaxialMaterial Hysteretic	5242	[lindex $OvZ   23 0]		  [expr [lindex $OvZ   23 1]*$sfthy] [lindex $OvZ   23 2] [lindex $OvZ   23 3]               [lindex $OvZ   23 4] [lindex $OvZ   23 5]               [lindex $OvZ   23 6]				     [expr [lindex $OvZ   23 7]*$sfthy] 			[lindex $OvZ   23 8] [lindex $OvZ   23 9]			 [lindex $OvZ   23 10] [lindex $OvZ   23 11] 			$pincXs	$pincYs	$d1	$d2sMS	$betasm

#                      tag    mat1 code1 mat2 code2 ...
section Aggregator     600   601    My  602    Mz  998 T 999  P 999  Vy 999  Vz
section Aggregator     5600  5601   My  5602   Mz  998 T 999  P 999  Vy 999  Vz
					 	  	
section Aggregator     150   151    My  152    Mz  998 T 999  P 999  Vy 999  Vz
section Aggregator     5150  5151   My  5152   Mz  998 T 999  P 999  Vy 999  Vz
					 	  	 
section Aggregator     270   241    My  242    Mz  998 T 999  P 999  Vy 999  Vz
section Aggregator     5270  5241   My  5242   Mz  998 T 999  P 999  Vy 999  Vz

####### beams
set sfthg 1.0;
                                         #M1p               th1p                             m2p                th2p                             m3p                th3p                             m1m                th1m                              m2m               th2m                            m3m                 th3m
uniaxialMaterial Hysteretic	1001	 [lindex $OvG  0 0] [expr [lindex $OvG  0 1]*$sfthg] [lindex $OvG  0 2] [expr [lindex $OvG  0 3]*$sfthg] [lindex $OvG  0 4] [expr [lindex $OvG  0 5]*$sfthg] [lindex $OvG  0 6] [expr [lindex $OvG  0 7]*$sfthg] [lindex $OvG  0 8] [expr [lindex $OvG  0 9]*$sfthg] [lindex $OvG  0 10] [expr [lindex $OvG  0 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	1002	 [lindex $OvG  1 0] [expr [lindex $OvG  1 1]*$sfthg] [lindex $OvG  1 2] [expr [lindex $OvG  1 3]*$sfthg] [lindex $OvG  1 4] [expr [lindex $OvG  1 5]*$sfthg] [lindex $OvG  1 6] [expr [lindex $OvG  1 7]*$sfthg] [lindex $OvG  1 8] [expr [lindex $OvG  1 9]*$sfthg] [lindex $OvG  1 10] [expr [lindex $OvG  1 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	2001	 [lindex $OvG  2 0] [expr [lindex $OvG  2 1]*$sfthg] [lindex $OvG  2 2] [expr [lindex $OvG  2 3]*$sfthg] [lindex $OvG  2 4] [expr [lindex $OvG  2 5]*$sfthg] [lindex $OvG  2 6] [expr [lindex $OvG  2 7]*$sfthg] [lindex $OvG  2 8] [expr [lindex $OvG  2 9]*$sfthg] [lindex $OvG  2 10] [expr [lindex $OvG  2 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	2002	 [lindex $OvG  3 0] [expr [lindex $OvG  3 1]*$sfthg] [lindex $OvG  3 2] [expr [lindex $OvG  3 3]*$sfthg] [lindex $OvG  3 4] [expr [lindex $OvG  3 5]*$sfthg] [lindex $OvG  3 6] [expr [lindex $OvG  3 7]*$sfthg] [lindex $OvG  3 8] [expr [lindex $OvG  3 9]*$sfthg] [lindex $OvG  3 10] [expr [lindex $OvG  3 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	3001	 [lindex $OvG  4 0] [expr [lindex $OvG  4 1]*$sfthg] [lindex $OvG  4 2] [expr [lindex $OvG  4 3]*$sfthg] [lindex $OvG  4 4] [expr [lindex $OvG  4 5]*$sfthg] [lindex $OvG  4 6] [expr [lindex $OvG  4 7]*$sfthg] [lindex $OvG  4 8] [expr [lindex $OvG  4 9]*$sfthg] [lindex $OvG  4 10] [expr [lindex $OvG  4 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	3002	 [lindex $OvG  5 0] [expr [lindex $OvG  5 1]*$sfthg] [lindex $OvG  5 2] [expr [lindex $OvG  5 3]*$sfthg] [lindex $OvG  5 4] [expr [lindex $OvG  5 5]*$sfthg] [lindex $OvG  5 6] [expr [lindex $OvG  5 7]*$sfthg] [lindex $OvG  5 8] [expr [lindex $OvG  5 9]*$sfthg] [lindex $OvG  5 10] [expr [lindex $OvG  5 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	4001	 [lindex $OvG  6 0] [expr [lindex $OvG  6 1]*$sfthg] [lindex $OvG  6 2] [expr [lindex $OvG  6 3]*$sfthg] [lindex $OvG  6 4] [expr [lindex $OvG  6 5]*$sfthg] [lindex $OvG  6 6] [expr [lindex $OvG  6 7]*$sfthg] [lindex $OvG  6 8] [expr [lindex $OvG  6 9]*$sfthg] [lindex $OvG  6 10] [expr [lindex $OvG  6 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	4002	 [lindex $OvG  7 0] [expr [lindex $OvG  7 1]*$sfthg] [lindex $OvG  7 2] [expr [lindex $OvG  7 3]*$sfthg] [lindex $OvG  7 4] [expr [lindex $OvG  7 5]*$sfthg] [lindex $OvG  7 6] [expr [lindex $OvG  7 7]*$sfthg] [lindex $OvG  7 8] [expr [lindex $OvG  7 9]*$sfthg] [lindex $OvG  7 10] [expr [lindex $OvG  7 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	5001	 [lindex $OvG  8 0] [expr [lindex $OvG  8 1]*$sfthg] [lindex $OvG  8 2] [expr [lindex $OvG  8 3]*$sfthg] [lindex $OvG  8 4] [expr [lindex $OvG  8 5]*$sfthg] [lindex $OvG  8 6] [expr [lindex $OvG  8 7]*$sfthg] [lindex $OvG  8 8] [expr [lindex $OvG  8 9]*$sfthg] [lindex $OvG  8 10] [expr [lindex $OvG  8 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	5002	 [lindex $OvG  9 0] [expr [lindex $OvG  9 1]*$sfthg] [lindex $OvG  9 2] [expr [lindex $OvG  9 3]*$sfthg] [lindex $OvG  9 4] [expr [lindex $OvG  9 5]*$sfthg] [lindex $OvG  9 6] [expr [lindex $OvG  9 7]*$sfthg] [lindex $OvG  9 8] [expr [lindex $OvG  9 9]*$sfthg] [lindex $OvG  9 10] [expr [lindex $OvG  9 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	6001	 [lindex $OvG 10 0] [expr [lindex $OvG 10 1]*$sfthg] [lindex $OvG 10 2] [expr [lindex $OvG 10 3]*$sfthg] [lindex $OvG 10 4] [expr [lindex $OvG 10 5]*$sfthg] [lindex $OvG 10 6] [expr [lindex $OvG 10 7]*$sfthg] [lindex $OvG 10 8] [expr [lindex $OvG 10 9]*$sfthg] [lindex $OvG 10 10] [expr [lindex $OvG 10 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	6002	 [lindex $OvG 11 0] [expr [lindex $OvG 11 1]*$sfthg] [lindex $OvG 11 2] [expr [lindex $OvG 11 3]*$sfthg] [lindex $OvG 11 4] [expr [lindex $OvG 11 5]*$sfthg] [lindex $OvG 11 6] [expr [lindex $OvG 11 7]*$sfthg] [lindex $OvG 11 8] [expr [lindex $OvG 11 9]*$sfthg] [lindex $OvG 11 10] [expr [lindex $OvG 11 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	7001	 [lindex $OvG 12 0] [expr [lindex $OvG 12 1]*$sfthg] [lindex $OvG 12 2] [expr [lindex $OvG 12 3]*$sfthg] [lindex $OvG 12 4] [expr [lindex $OvG 12 5]*$sfthg] [lindex $OvG 12 6] [expr [lindex $OvG 12 7]*$sfthg] [lindex $OvG 12 8] [expr [lindex $OvG 12 9]*$sfthg] [lindex $OvG 12 10] [expr [lindex $OvG 12 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	7002	 [lindex $OvG 13 0] [expr [lindex $OvG 13 1]*$sfthg] [lindex $OvG 13 2] [expr [lindex $OvG 13 3]*$sfthg] [lindex $OvG 13 4] [expr [lindex $OvG 13 5]*$sfthg] [lindex $OvG 13 6] [expr [lindex $OvG 13 7]*$sfthg] [lindex $OvG 13 8] [expr [lindex $OvG 13 9]*$sfthg] [lindex $OvG 13 10] [expr [lindex $OvG 13 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	8001	 [lindex $OvG 14 0] [expr [lindex $OvG 14 1]*$sfthg] [lindex $OvG 14 2] [expr [lindex $OvG 14 3]*$sfthg] [lindex $OvG 14 4] [expr [lindex $OvG 14 5]*$sfthg] [lindex $OvG 14 6] [expr [lindex $OvG 14 7]*$sfthg] [lindex $OvG 14 8] [expr [lindex $OvG 14 9]*$sfthg] [lindex $OvG 14 10] [expr [lindex $OvG 14 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	8002	 [lindex $OvG 15 0] [expr [lindex $OvG 15 1]*$sfthg] [lindex $OvG 15 2] [expr [lindex $OvG 15 3]*$sfthg] [lindex $OvG 15 4] [expr [lindex $OvG 15 5]*$sfthg] [lindex $OvG 15 6] [expr [lindex $OvG 15 7]*$sfthg] [lindex $OvG 15 8] [expr [lindex $OvG 15 9]*$sfthg] [lindex $OvG 15 10] [expr [lindex $OvG 15 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	9001	 [lindex $OvG 16 0] [expr [lindex $OvG 16 1]*$sfthg] [lindex $OvG 16 2] [expr [lindex $OvG 16 3]*$sfthg] [lindex $OvG 16 4] [expr [lindex $OvG 16 5]*$sfthg] [lindex $OvG 16 6] [expr [lindex $OvG 16 7]*$sfthg] [lindex $OvG 16 8] [expr [lindex $OvG 16 9]*$sfthg] [lindex $OvG 16 10] [expr [lindex $OvG 16 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	9002	 [lindex $OvG 17 0] [expr [lindex $OvG 17 1]*$sfthg] [lindex $OvG 17 2] [expr [lindex $OvG 17 3]*$sfthg] [lindex $OvG 17 4] [expr [lindex $OvG 17 5]*$sfthg] [lindex $OvG 17 6] [expr [lindex $OvG 17 7]*$sfthg] [lindex $OvG 17 8] [expr [lindex $OvG 17 9]*$sfthg] [lindex $OvG 17 10] [expr [lindex $OvG 17 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	10001	 [lindex $OvG 18 0] [expr [lindex $OvG 18 1]*$sfthg] [lindex $OvG 18 2] [expr [lindex $OvG 18 3]*$sfthg] [lindex $OvG 18 4] [expr [lindex $OvG 18 5]*$sfthg] [lindex $OvG 18 6] [expr [lindex $OvG 18 7]*$sfthg] [lindex $OvG 18 8] [expr [lindex $OvG 18 9]*$sfthg] [lindex $OvG 18 10] [expr [lindex $OvG 18 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	10002	 [lindex $OvG 19 0] [expr [lindex $OvG 19 1]*$sfthg] [lindex $OvG 19 2] [expr [lindex $OvG 19 3]*$sfthg] [lindex $OvG 19 4] [expr [lindex $OvG 19 5]*$sfthg] [lindex $OvG 19 6] [expr [lindex $OvG 19 7]*$sfthg] [lindex $OvG 19 8] [expr [lindex $OvG 19 9]*$sfthg] [lindex $OvG 19 10] [expr [lindex $OvG 19 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	11001	 [lindex $OvG 20 0] [expr [lindex $OvG 20 1]*$sfthg] [lindex $OvG 20 2] [expr [lindex $OvG 20 3]*$sfthg] [lindex $OvG 20 4] [expr [lindex $OvG 20 5]*$sfthg] [lindex $OvG 20 6] [expr [lindex $OvG 20 7]*$sfthg] [lindex $OvG 20 8] [expr [lindex $OvG 20 9]*$sfthg] [lindex $OvG 20 10] [expr [lindex $OvG 20 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	11002	 [lindex $OvG 21 0] [expr [lindex $OvG 21 1]*$sfthg] [lindex $OvG 21 2] [expr [lindex $OvG 21 3]*$sfthg] [lindex $OvG 21 4] [expr [lindex $OvG 21 5]*$sfthg] [lindex $OvG 21 6] [expr [lindex $OvG 21 7]*$sfthg] [lindex $OvG 21 8] [expr [lindex $OvG 21 9]*$sfthg] [lindex $OvG 21 10] [expr [lindex $OvG 21 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	12001	 [lindex $OvG 22 0] [expr [lindex $OvG 22 1]*$sfthg] [lindex $OvG 22 2] [expr [lindex $OvG 22 3]*$sfthg] [lindex $OvG 22 4] [expr [lindex $OvG 22 5]*$sfthg] [lindex $OvG 22 6] [expr [lindex $OvG 22 7]*$sfthg] [lindex $OvG 22 8] [expr [lindex $OvG 22 9]*$sfthg] [lindex $OvG 22 10] [expr [lindex $OvG 22 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	12002	 [lindex $OvG 23 0] [expr [lindex $OvG 23 1]*$sfthg] [lindex $OvG 23 2] [expr [lindex $OvG 23 3]*$sfthg] [lindex $OvG 23 4] [expr [lindex $OvG 23 5]*$sfthg] [lindex $OvG 23 6] [expr [lindex $OvG 23 7]*$sfthg] [lindex $OvG 23 8] [expr [lindex $OvG 23 9]*$sfthg] [lindex $OvG 23 10] [expr [lindex $OvG 23 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	13001	 [lindex $OvG 24 0] [expr [lindex $OvG 24 1]*$sfthg] [lindex $OvG 24 2] [expr [lindex $OvG 24 3]*$sfthg] [lindex $OvG 24 4] [expr [lindex $OvG 24 5]*$sfthg] [lindex $OvG 24 6] [expr [lindex $OvG 24 7]*$sfthg] [lindex $OvG 24 8] [expr [lindex $OvG 24 9]*$sfthg] [lindex $OvG 24 10] [expr [lindex $OvG 24 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	13002	 [lindex $OvG 25 0] [expr [lindex $OvG 25 1]*$sfthg] [lindex $OvG 25 2] [expr [lindex $OvG 25 3]*$sfthg] [lindex $OvG 25 4] [expr [lindex $OvG 25 5]*$sfthg] [lindex $OvG 25 6] [expr [lindex $OvG 25 7]*$sfthg] [lindex $OvG 25 8] [expr [lindex $OvG 25 9]*$sfthg] [lindex $OvG 25 10] [expr [lindex $OvG 25 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	14001	 [lindex $OvG 26 0] [expr [lindex $OvG 26 1]*$sfthg] [lindex $OvG 26 2] [expr [lindex $OvG 26 3]*$sfthg] [lindex $OvG 26 4] [expr [lindex $OvG 26 5]*$sfthg] [lindex $OvG 26 6] [expr [lindex $OvG 26 7]*$sfthg] [lindex $OvG 26 8] [expr [lindex $OvG 26 9]*$sfthg] [lindex $OvG 26 10] [expr [lindex $OvG 26 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
uniaxialMaterial Hysteretic	14002	 [lindex $OvG 27 0] [expr [lindex $OvG 27 1]*$sfthg] [lindex $OvG 27 2] [expr [lindex $OvG 27 3]*$sfthg] [lindex $OvG 27 4] [expr [lindex $OvG 27 5]*$sfthg] [lindex $OvG 27 6] [expr [lindex $OvG 27 7]*$sfthg] [lindex $OvG 27 8] [expr [lindex $OvG 27 9]*$sfthg] [lindex $OvG 27 10] [expr [lindex $OvG 27 11]*$sfthg]	$pincXg	$pincYg	$d1	$d2g	$betag
																        				 
#		       tag    mat1 code1 											        				 
section Aggregator     1001   1001  My  999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     1002   1002  My  999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     2001   2001  My  999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     2002   2002  My  999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     3001   3001  My  999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     3002   3002  My  999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     4001   4001  My  999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     4002   4002  My	999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     5001   5001  My	999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     5002   5002  My	999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     6001   6001  My	999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     6002   6002  My	999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     7001   7001  My	999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     7002   7002  My	999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     8001   8001  My	999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     8002   8002  My	999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     9001   9001  My	999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     9002   9002  My	999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     10001  10001 My	999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     10002  10002 My	999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     11001  11001 My	999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     11002  11002 My	999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     12001  12001 My	999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     12002  12002 My	999  Mz 999 T 999  P 999  Vy 999  Vz
						  
section Aggregator     13001  13001 My	999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     13002  13002 My	999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     14001  14001 My	999  Mz 999 T 999  P 999  Vy 999  Vz
section Aggregator     14002  14002 My	999  Mz 999 T 999  P 999  Vy 999  Vz









































































































































































































































































































































































































































































































































