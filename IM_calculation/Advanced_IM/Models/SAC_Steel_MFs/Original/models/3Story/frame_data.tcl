##############################################################################################################
# Reagan Chandramohan                                                                                        #
# John A. Blume Earthquake Engineering Center                                                                #
# Stanford University                                                                                        #
# Last edited: 27-May-2014                                                                                   #
##############################################################################################################

# Input data file used by "create_steel_mf_model.tcl" to create a non-linear, concentrated plasticity
# model of a steel moment frame.
# All units in kip and in.

##############################################################################################################

# Number of bays and list of bay widths
set num_bays 4
set bay_widths [lrepeat 4 [expr {30.0*$ft}]]

# Number of stories and list of story heights (inclusive of basement levels if any)
set num_stories 3
set story_heights [lrepeat 3 [expr {13.0*$ft}]]

# Flag to indicate if basement levels are present and the number of basement levels (set to 0 if none)
set basement_flag false
set num_basement_levels 0

# List of lists containing column cross section shapes, going from left to right and bottom to top
set col_shapes "
    {W14X342 [lrepeat 3 W14X398] W14X342}
    {W14X342,W14X159 [lrepeat 3 W14X398,W14X159] W14X342,W14X159}
    {W14X159 [lrepeat 3 W14X159] W14X159}
"

# Height of column splice from beam centerline if column splices are to be modeled
set splice_height [expr {5.0*$ft}]

# List of lists containing beam cross section shapes, going from left to right and bottom to top
set beam_shapes "
    {[lrepeat 4 W33X141]}
    {[lrepeat 4 W21X62]}
    {[lrepeat 4 W21X62]}
"

# Flag that defines how the unbraced lengths of the beams are to be computed
# 1: Values are specified for each beam in a list called beam_Lb, going from left to right and bottom to top
# 2: Computed as the distance between the inner column faces (no additional bracing)
# 3: Computed based on the requirements for a special steel moment frame in AISC 341
# 4: Computed based on the requirements for an intermediate steel moment frame in AISC 341
set beam_Lb_flag 2
#set beam_Lb {}

# List of lists containing column axial forces from a first order analysis (in Mastan2), going from left
# to right and bottom to top. Required to compute yield moments of columns using an interaction curve.
set col_axforce {
    {76.24 151.7 152.1 151.7 76.24}
    {50.02 99.64 99.82 99.64 50.02}
    {23.86 47.51 47.62 47.51 23.86}
}

# List of lists containing doubler plate thicknesses, going from left to right and bottom to top
set doubler_plate_thicknesses "
    {[lrepeat 5 0.375]}
    {[lrepeat 5 0.625]}
    {[lrepeat 5 0.875]}
"

# Flags specifying whether or not beams have RBS cuts and/or cover plates on the flanges at beam-column joints
set beam_rbs_flag false
set beam_cover_plate_flag true

# If RBS cuts or cover plates are present, specify hinge offsets from the column face for all stories and bays 
# going from left to right and bottom to top
set beam_hinge_offsets "
    {[lrepeat 8 17.0]}
    {[lrepeat 8 11.0]}
    {[lrepeat 8 11.0]}
"

# If RBS cuts are present, specify the minimum flange widths in the RBS cuts
#set rbs_widths {}

# If cover plates are present, specify top and bottom cover plate widths and thicknesses
dict set cover_plates widths top "
    {[lrepeat 8 11.5]}
    {[lrepeat 8 8.25]}
    {[lrepeat 8 8.25]}
"

dict set cover_plates widths bottom "
    {[lrepeat 8 13.5]}
    {[lrepeat 8 10.25]}
    {[lrepeat 8 10.25]}
"

dict set cover_plates thicknesses top "
    {[lrepeat 8 1.125]}
    {[lrepeat 8 0.75]}
    {[lrepeat 8 0.75]}
"

dict set cover_plates thicknesses bottom "
    {[lrepeat 8 1.0]}
    {[lrepeat 8 0.625]}
    {[lrepeat 8 0.625]}
"

# List of total seismic mass at each story, including basement levels if any
# Mass assigned to the basement level is inconsequential, but is required for model stability
set story_masses "[lrepeat 2 [expr {65.53/(12.0*2.0)}]] [expr {70.90/(12.0*2.0)}]"

# Flag that defines how the gravity loads at each story are to be computed
# 1: Computed from the story masses
# 2: Values are specified for each story, including basement levels if any, in the "story_loads" list
set story_load_flag 2
set story_loads "[lrepeat 2 [expr {(96.0 + 20.0)*24.0*30.0*30.0/(2.0*1e3)}]] \
                 [expr {((83.0 + 20.0)*22.0 + (116.0 + 20.0)*2.0)*30.0*30.0/(2.0*1e3)}]"

# List of the fraction of the seismic mass and gravity load tributary directly to the frame
# (the remaining is assumed to be tributary to the leaning column)
set tributary_ratios_frame [lrepeat 3 [expr {1.0/6.0}]]

# List of the fractions of the area tributary to the frame that are tributary to each individual column
# Same values assumed for all stories
set tributary_ratios_columns "[expr {1.0/$num_bays/2.0}] [lrepeat 3 [expr {1.0/$num_bays}]] \
        [expr {1.0/$num_bays/2.0}]"

	# puts "data is done!"