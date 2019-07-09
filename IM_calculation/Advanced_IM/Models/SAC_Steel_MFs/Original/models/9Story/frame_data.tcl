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
set num_stories 10
set story_heights "[expr {12.0*$ft}] [expr {18.0*$ft}] [lrepeat 8 [expr {13.0*$ft}]]"

# Flag to indicate if basement levels are present. If present, specify the number of basement levels.
set basement_flag true
set num_basement_levels 1

# List of lists containing column cross section shapes, going from left to right and bottom to top
set col_shapes "
    [lrepeat 4 "W24X176 [lrepeat 3 W24X229] W24X176"]
    {W24X176 [lrepeat 3 W24X229,W24X207] W24X176}
    {W24X176 [lrepeat 3 W24X207] W24X176}
    {W24X176,W24X131 [lrepeat 3 W24X207,W24X162] W24X176,W24X131}
    {W24X131 [lrepeat 3 W24X162] W24X131}
    {W24X131,W24X94 [lrepeat 3 W24X162,W24X131] W24X131,W24X94}
    {W24X94 [lrepeat 3 W24X131] W24X94}
"

# Height of column splice from beam centerline if column splices are to be modeled
set splice_height [expr {5.0*$ft}]

# List of lists containing beam cross section shapes, going from left to right and bottom to top
set beam_shapes "
    {[lrepeat 4 W30X108]}
    [lrepeat 2 [lrepeat 4 W30X116]]
    {[lrepeat 4 W30X108]}
    [lrepeat 2 [lrepeat 4 W27X94]]
    {[lrepeat 4 W24X84]}
    {[lrepeat 4 W24X76]}
    {[lrepeat 4 W24X55]}
    {[lrepeat 4 W21X44]}
"

# Flag that defines how the unbraced lengths of the beams are to be computed
# 1: Values are specified for each beam in a list called beam_Lb, going from left to right and bottom to top
# 2: Computed as the distance between the inner column faces (no additional bracing)
# 3: Computed based on the requirements for a special steel moment frame in AISC 341
# 4: Computed based on the requirements for an intermediate steel moment frame in AISC 341
set beam_Lb_flag 3
#set beam_Lb {}

# List of lists containing column axial forces from a first order analysis (in Mastan2), going from left
# to right and bottom to top. Required to compute yield moments of columns using an interaction curve.
set col_axforce {
    {514.0 524.7 517.3 508.2 263.9}
    {462.0 472.2 465.1 456.3 237.5}
    {410.1 419.3 412.9 404.9 211.0}
    {358.4 366.2 360.7 353.8 184.3}
    {306.7 312.9 308.5 302.8 157.5}
    {254.9 259.7 256.4 251.8 130.8}
    {203.1 206.6 204.1 200.9 103.9}
    {151.2 153.6 151.9 149.7 77.23}
    {99.36 100.6 99.74 98.57 50.54}
    {47.39 47.87 47.54 47.09 24.02}
}

# List of lists containing doubler plate thicknesses, going from left to right and bottom to top
set doubler_plate_thicknesses "
    [lrepeat 3 [lrepeat 5 0.25]]
    [lrepeat 3 [lrepeat 5 0.0]]
    {[lrepeat 5 0.25]}
    [lrepeat 3 [lrepeat 5 0.0]]
"

# Flags specifying whether or not beams have RBS cuts and/or cover plates on the flanges at beam-column joints
set beam_rbs_flag true
set beam_cover_plate_flag true

# If RBS cuts or cover plates are present, specify hinge offsets from the column face for all stories and bays 
# going from left to right and bottom to top
set beam_hinge_offsets "
    {[lrepeat 8 26.5]}
    [lrepeat 2 [lrepeat 8 26.5]]
    {[lrepeat 8 26.5]}
    [lrepeat 2 [lrepeat 8 24.0]]
    {[lrepeat 8 21.0]}
    {[lrepeat 8 21.0]}
    {[lrepeat 8 21.0]}
    {[lrepeat 8 18.5]}
"

# If RBS cuts are present, specify the minimum flange widths in the RBS cuts
set rbs_widths "
    {[lrepeat 8 5.25]}
    [lrepeat 2 [lrepeat 8 5.25]]
    {[lrepeat 8 5.25]}
    [lrepeat 2 [lrepeat 8 5.0]]
    {[lrepeat 8 5.0]}
    {[lrepeat 8 4.5]}
    {[lrepeat 8 3.5]}
    {[lrepeat 8 3.25]}
"

# If cover plates are present, specify top and bottom cover plate widths and thicknesses
dict set cover_plates widths top "
    {[lrepeat 8 10.5]}
    [lrepeat 2 [lrepeat 8 10.5]]
    {[lrepeat 8 10.5]}
    [lrepeat 2 [lrepeat 8 10.0]]
    {[lrepeat 8 9.0]}
    {[lrepeat 8 9.0]}
    {[lrepeat 8 7.0]}
    {[lrepeat 8 6.5]}
"

dict set cover_plates widths bottom "
    {[lrepeat 8 12.5]}
    [lrepeat 2 [lrepeat 8 12.5]]
    {[lrepeat 8 12.5]}
    [lrepeat 2 [lrepeat 8 12.0]]
    {[lrepeat 8 11.0]}
    {[lrepeat 8 11.0]}
    {[lrepeat 8 9.0]}
    {[lrepeat 8 8.5]}
"

dict set cover_plates thicknesses top "
    {[lrepeat 8 0.4375]}
    [lrepeat 2 [lrepeat 8 0.4375]]
    {[lrepeat 8 0.4375]}
    [lrepeat 2 [lrepeat 8 0.375]]
    {[lrepeat 8 0.375]}
    {[lrepeat 8 0.3125]}
    {[lrepeat 8 0.375]}
    {[lrepeat 8 0.375]}
"

dict set cover_plates thicknesses bottom "
    {[lrepeat 8 0.375]}
    [lrepeat 2 [lrepeat 8 0.375]]
    {[lrepeat 8 0.375]}
    [lrepeat 2 [lrepeat 8 0.3125]]
    {[lrepeat 8 0.3125]}
    {[lrepeat 8 0.25]}
    {[lrepeat 8 0.3125]}
    {[lrepeat 8 0.25]}
"

# List of total seismic mass at each story, including basement levels if any
# Mass assigned to the basement level is inconsequential, but is required for model stability
set story_masses "[lrepeat 2 [expr {69.04/(12.0*2.0)}]] [lrepeat 7 [expr {67.86/(12.0*2.0)}]] \
        [expr {73.10/(12.0*2.0)}]"

# Flag that defines how the gravity loads at each story are to be computed
# 1: Computed from the story masses
# 2: Values are specified for each story, including basement levels if any, in the "story_loads" list
set story_load_flag 2
set story_loads "[lrepeat 9 [expr {(96.0 + 20.0)*25.0*30.0*30.0/(2.0*1e3)}]] \
                 [expr {((83.0 + 20.0)*23.0 + (116.0 + 20.0)*2.0)*30.0*30.0/(2.0*1e3)}]"

# List of the fraction of the seismic mass and gravity load tributary directly to the frame
# (the remaining is assumed to be tributary to the leaning column)
set tributary_ratios_frame [lrepeat 10 [expr {9.0/50.0}]]

# List of the fractions of the area tributary to the frame that are tributary to each individual column
# Same values assumed for all stories
set tributary_ratios_columns "[lrepeat 4 [expr {2.0/9.0}]] [expr {1.0/9.0}]"
