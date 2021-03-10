
set FEMs_list "SCBF_6Story"  
set GM_name  "ChCh22Feb2011_Obs"
set dir 000
set iTry_fem 0	
set gm_list "REHS" 
set iTry_gm 0
set output "output_nw_cd"
# Loop over all the model directories
foreach FEMs $FEMs_list {
		     set FEMs [lindex $FEMs_list $iTry_fem]


set modelpath Models
set modeldir $modelpath/$FEMs


puts "Start Analyzing..."
set model_name $FEMs

set station_num 1

foreach gm $gm_list {
                      set gm [lindex $gm_list $iTry_gm]


foreach gMotion [glob -nocomplain -directory ../GMotions3/$GM_name $gm.$dir] {     
  
         set station_path [string range $gMotion 0 end-0]
         set station_name [string range $gMotion 30 end-4]	
         # puts "station_path = $station_path"		 
	 puts "GM=$GM_name"	
	 puts "stations number = $station_num"
         puts "FEMs=$FEMs"
         puts " "
	 puts "-------------------------------"
	 puts " "
	 
       	source general/GMs.tcl
	# puts "Station no= $iRec"	
	# puts "Type= $type_GM"
	puts " "
	source $modeldir/build_modified_atc84_model.tcl
	source general/Period.tcl
	source  general/Recorders.tcl
	source general/Rha_nm_cd.tcl
	
	puts " "
	puts "---------------------------------------------------------------------------"
	puts " "
	incr station_num		
        
    wipe
	
	       
}	

incr iTry_gm
unset status_nw
unset station
# unset status_cd
}
incr iTry_fem
}

