##############################################################################################################
# Reagan Chandramohan                                                                                        #
# John A. Blume Earthquake Engineering Center                                                                #
# Stanford University                                                                                        #
# Last edited: 07-Oct-2014                                                                                   #
##############################################################################################################

# Create a non-linear, concentrated plasticity model of a steel moment frame using information from
# "model_data_file"

proc CreateSteelMFModel {model_data_file FEMs} {
    
     	
    # Source required files
    source [file join [file dirname [info script]] Original/constants_units_kip_in.tcl]
    source [file join [file dirname [info script]] Original/steel_wshapes.tcl]
    source [file join [file dirname [info script]] Original/atc72_equations.tcl]
	
    source $model_data_file
    
      model BasicBuilder -ndm 2

    # Set debug flag to true to create a file with information about each plastic hinge
    set debug_flag false

    ############################################  Create Nodes  ############################################

    # Create lists of x and y coordinates of all beam-column joints
    set x_coords 0.0
    foreach bay_width $bay_widths {
        lappend x_coords [expr {[lindex $x_coords end] + $bay_width}]
    }

    set y_coords 0.0
    foreach story_height $story_heights {
        lappend y_coords [expr {[lindex $y_coords end] + $story_height}]
    }

    # Loop over all bays
    for {set bay 0} {$bay <= $num_bays} {incr bay} {
        
        # If there is no basement, place two nodes at ground level and fix one, else place one node and
        # pin it
        set x_coord [lindex $x_coords $bay]
        if {$basement_flag} {
            node 1[format "%02d" $bay]002 $x_coord 0.0

            fix 1[format "%02d" $bay]002 1 1 0
        } else {
            node 1[format "%02d" $bay]004 $x_coord 0.0
            node 1[format "%02d" $bay]002 $x_coord 0.0

            fix 1[format "%02d" $bay]004 1 1 1
        }

        # Loop over all stories to create nodes at each beam-column joint
        for {set story 1} {$story <= $num_stories} {incr story} {
            
            # Get the node offsets in the x direction based on the depth of the column below the joint
            # Account for the fact that the column could contain a splice
            set col_sections [split [lindex [lindex $col_shapes [expr {$story - 1}]] $bay] ,]
            if {[llength $col_sections] == 1} {
                set col_section $col_sections
            } else {
                set col_section [lindex $col_sections 1]
            }
            set col_depth [dict get $wshape_props $col_section d]
            set x_offset [expr {$col_depth/2}]

            # Get the node offsets in the y direction based on the maximum depth of the two beams framing in
            if {$bay > 0} {
                set beam_section [lindex [lindex $beam_shapes [expr {$story - 1}]] [expr {$bay - 1}]]
                set beam_depth_left [dict get $wshape_props $beam_section d]
            } else {
                set beam_depth_left 0.0
            }
            if {$bay < $num_bays} {
                set beam_section [lindex [lindex $beam_shapes [expr {$story - 1}]] $bay]
                set beam_depth_right [dict get $wshape_props $beam_section d]
            } else {
                set beam_depth_right 0.0
            }
            set y_offset [expr {max($beam_depth_left, $beam_depth_right)/2}]

            # Place nodes around panel zone of beam-column joint
            set y_coord [lindex $y_coords $story]
            node 1[format "%02d" $bay][format "%02d" $story]1 [expr {$x_coord + $x_offset}] $y_coord
            node 1[format "%02d" $bay][format "%02d" $story]2 $x_coord [expr {$y_coord + $y_offset}]
            node 1[format "%02d" $bay][format "%02d" $story]3 [expr {$x_coord - $x_offset}] $y_coord
            node 1[format "%02d" $bay][format "%02d" $story]4 $x_coord [expr {$y_coord - $y_offset}]

            # If the story has a column splice, place a node at the location of the splice
            if {[llength $col_sections] != 1} {
                node 3[format "%02d" $bay][format "%02d" $story] $x_coord \
                        [expr {[lindex $y_coords [expr {$story - 1}]] + $splice_height}]
            }

            # If beams have RBS cuts or cover plates, place two nodes to the left and two to the right of
            # the panel zone at the defined offsets from the column faces
            if {$beam_rbs_flag || $beam_cover_plate_flag} {
                if {$bay > 0} {
                    set beam_hinge_offset_left [lindex [lindex $beam_hinge_offsets [expr {$story - 1}]] \
                            [expr {2*$bay - 1}]]
                    node 1[format "%02d" $bay][format "%02d" $story]5 \
                            [expr {$x_coord - $x_offset - $beam_hinge_offset_left}] $y_coord
                    node 1[format "%02d" $bay][format "%02d" $story]6 \
                            [expr {$x_coord - $x_offset - $beam_hinge_offset_left}] $y_coord
                }
                if {$bay < $num_bays} {
                    set beam_hinge_offset_right [lindex [lindex $beam_hinge_offsets [expr {$story - 1}]] \
                            [expr {2*$bay}]]
                    node 1[format "%02d" $bay][format "%02d" $story]7 \
                            [expr {$x_coord + $x_offset + $beam_hinge_offset_right}] $y_coord
                    node 1[format "%02d" $bay][format "%02d" $story]8 \
                            [expr {$x_coord + $x_offset + $beam_hinge_offset_right}] $y_coord
                }
            }
        }
    }

    # If the number of bays is odd, create a node at each story in the middle of the center bay to connect
    # to the leaning column
    set x_coord [expr {[lindex $x_coords end]/2.0}]
    if {$num_bays % 2} {
        for {set story 1} {$story <= $num_stories} {incr story} {
            set y_coord [lindex $y_coords $story]
            node 2[format "%02d" $story] $x_coord $y_coord
        }
    }

    # Loop over all basement levels, create nodes at a unit distance from either side of the exterior
    # panel zones, and fix them
    if {$basement_flag} {
        for {set story 1} {$story <= $num_basement_levels} {incr story} {
            node 400[format "%02d" $story]0 [expr {[nodeCoord 100[format "%02d" $story]3 1] - 1.0}] \
                    [lindex $y_coords $story]
            node 4[format "%02d" $num_bays][format "%02d" $story]0 \
                    [expr {[nodeCoord 1[format "%02d" $num_bays][format "%02d" $story]1 1] + 1.0}] \
                    [lindex $y_coords $story]

            fix 400[format "%02d" $story]0 1 1 1
            fix 4[format "%02d" $num_bays][format "%02d" $story]0 1 1 1
        }
    }
# puts "node was created!"
    #######################################  Create Elastic Elements  #######################################

    # Define geometric transformation
    set pdelta 1
    geomTransf PDelta $pdelta

    # Define stiffness distribution factor
    set n 10.0

    # Keep track of largest cross sectional area and moment of inertia of all elements in the frame to be
    # later used to define the properties of the leaning column
    set A_max 0.0
    set I_max 0.0

    # Loop and create all  columns
    for {set bay 0} {$bay <= $num_bays} {incr bay} {
        for {set story 1} {$story <= $num_stories} {incr story} {

            # Check if the column contains a splice and create one or two elements accordingly
            set col_sections [split [lindex [lindex $col_shapes [expr {$story - 1}]] $bay] ,]

            if {[llength $col_sections] == 1} {
                set col_section $col_sections

                # Get column cross section properties
                set A [dict get $wshape_props $col_section A]
                set I [dict get $wshape_props $col_section Ix]
                set I_mod [expr {($n + 1.0)/$n*$I}]

                # Update A_max and I_max
                set A_max [expr {max($A_max, $A)}]
                set I_max [expr {max($I_max, $I)}]

                # Keep track of the sum of cross sectional areas at all stories
                if {$bay == 0} {
                    set A_sum($story) $A
                } else {
                    set A_sum($story) [expr {$A_sum($story) + $A}]
                }
                
                # Identify nodei and nodej
                set nodei 1[format "%02d" $bay][format "%02d" [expr {$story - 1}]]2
                set nodej 1[format "%02d" $bay][format "%02d" $story]4

                # Compute the column length
                set col_lengths($bay,$story) [expr {[nodeCoord $nodej 2] - [nodeCoord $nodei 2]}]

                # Define elastic beam-column element
                element elasticBeamColumn 10[format "%02d" $bay][format "%02d" $story] $nodei $nodej $A $E \
                        $I_mod $pdelta
            } else {
                for {set col_num 0} {$col_num < 2} {incr col_num} {
                    set col_section [lindex $col_sections $col_num]
                    
                    # Get column cross section properties
                    set A [dict get $wshape_props $col_section A]
                    set I [dict get $wshape_props $col_section Ix]
                    set I_mod [expr {($n + 1.0)/$n*$I}]

                    # Update A_max and I_max
                    set A_max [expr {max($A_max, $A)}]
                    set I_max [expr {max($I_max, $I)}]

                    # Keep track of the sum of cross sectional areas at all stories
                    if {$col_num == 0} {
                        if {$bay == 0} {
                            set A_sum($story) $A
                        } else {
                            set A_sum($story) [expr {$A_sum($story) + $A}]
                        }
                    }
                    
                    # Identify nodei and nodej
                    if {$col_num == 0} {
                        set nodei 1[format "%02d" $bay][format "%02d" [expr {$story - 1}]]2
                        set nodej 3[format "%02d" $bay][format "%02d" $story]
                    } else {
                        set nodei 3[format "%02d" $bay][format "%02d" $story]
                        set nodej 1[format "%02d" $bay][format "%02d" $story]4
                    }

                    # Compute the column length
                    if {$col_num == 0} {
                        set col_lengths($bay,$story) 0.0
                    }
                    set col_lengths($bay,$story) [expr {$col_lengths($bay,$story) + [nodeCoord $nodej 2] - \
                            [nodeCoord $nodei 2]}]

                    # Define elastic beam-column element
                    if {$col_num == 0} {
                        set element_tag 11
                    } else {
                        set element_tag 12
                    }
                    element elasticBeamColumn $element_tag[format "%02d" $bay][format "%02d" $story] \
                            $nodei $nodej $A $E $I_mod $pdelta
                }
            }
        }
    }

    # Loop and create all beams
    for {set bay 0} {$bay < $num_bays} {incr bay} {
        for {set story 1} {$story <= $num_stories} {incr story} {
            
            # Get beam cross section properties
            set beam_section [lindex [lindex $beam_shapes [expr {$story - 1}]] $bay]
            set A [dict get $wshape_props $beam_section A]
            set I [dict get $wshape_props $beam_section Ix]
            set I_mod [expr {($n + 1.0)/$n*$I}]

            # If beam flages have cover plates installed at the column faces, compute an increment to the
            # moment of inertia of the cross section of the beam element from the joint panel to the plastic
            # hinge zone
            # Assume that the cross section centroid remains in the middle of the section even if the
            # top and bottom cover plates are of different dimensions
            # Note: "right" and "left" are with respect to the beam-column joint, not the beam
            if {$beam_cover_plate_flag} {
                set d [dict get $wshape_props $beam_section d]

                set cover_b_top [lindex [lindex [dict get $cover_plates widths top] \
                        [expr {$story - 1}]] [expr {2*$bay}]]
                set cover_b_bot [lindex [lindex [dict get $cover_plates widths bottom] \
                        [expr {$story - 1}]] [expr {2*$bay}]]
                set cover_t_top [lindex [lindex [dict get $cover_plates thicknesses top] \
                        [expr {$story - 1}]] [expr {2*$bay}]]
                set cover_t_bot [lindex [lindex [dict get $cover_plates thicknesses bottom] \
                        [expr {$story - 1}]] [expr {2*$bay}]]
                set cover_plate_right_Iinc [expr {$cover_b_top*($cover_t_top**3)/12.0 + \
                        $cover_b_top*$cover_t_top*(($d + $cover_t_top)/2.0)**2 + \
                        $cover_b_bot*($cover_t_bot**3)/12.0 + \
                        $cover_b_bot*$cover_t_bot*(($d + $cover_t_bot)/2.0)**2}]

                set cover_b_top [lindex [lindex [dict get $cover_plates widths top] \
                        [expr {$story - 1}]] [expr {2*$bay + 1}]]
                set cover_b_bot [lindex [lindex [dict get $cover_plates widths bottom] \
                        [expr {$story - 1}]] [expr {2*$bay + 1}]]
                set cover_t_top [lindex [lindex [dict get $cover_plates thicknesses top] \
                        [expr {$story - 1}]] [expr {2*$bay + 1}]]
                set cover_t_bot [lindex [lindex [dict get $cover_plates thicknesses bottom] \
                        [expr {$story - 1}]] [expr {2*$bay + 1}]]
                set cover_plate_left_Iinc [expr {$cover_b_top*($cover_t_top**3)/12.0 + \
                        $cover_b_top*$cover_t_top*(($d + $cover_t_top)/2.0)**2 + \
                        $cover_b_bot*($cover_t_bot**3)/12.0 + \
                        $cover_b_bot*$cover_t_bot*(($d + $cover_t_bot)/2.0)**2}]
            } else {
                set cover_plate_right_Iinc 0.0
                set cover_plate_left_Iinc 0.0
            }

            # Identify nodei and nodej
            set nodei 1[format "%02d" $bay][format "%02d" $story]1
            set nodej 1[format "%02d" [expr {$bay + 1}]][format "%02d" $story]3

            # Compute the beam length
            set beam_lengths($bay,$story) [expr {[nodeCoord $nodej 1] - [nodeCoord $nodei 1]}]

            # Define elastic beam-column element(s) and compute the hinge-to-hinge distances
            if {$beam_rbs_flag || $beam_cover_plate_flag} {
                set nodek1 1[format "%02d" $bay][format "%02d" $story]7
                set nodek2 1[format "%02d" $bay][format "%02d" $story]8

                set nodel1 1[format "%02d" [expr {$bay + 1}]][format "%02d" $story]5
                set nodel2 1[format "%02d" [expr {$bay + 1}]][format "%02d" $story]6

                element elasticBeamColumn 22[format "%02d" $bay][format "%02d" $story] $nodei $nodek1 $A $E \
                        [expr {$I_mod + $cover_plate_right_Iinc}] $pdelta
                element elasticBeamColumn 21[format "%02d" [expr {$bay + 1}]][format "%02d" $story] $nodel2 \
                        $nodej $A $E [expr {$I_mod + $cover_plate_left_Iinc}] $pdelta

                if {($num_bays % 2) && ($bay == $num_bays/2)} {
                    set nodem 2[format "%02d" $story]
                    element elasticBeamColumn 23[format "%02d" $bay][format "%02d" $story] $nodek2 $nodem \
                            $A $E $I_mod $pdelta
                    element elasticBeamColumn 23[format "%02d" [expr {$bay + 1}]][format "%02d" $story] \
                            $nodem $nodel1 $A $E $I_mod $pdelta
                } else {
                    element elasticBeamColumn 20[format "%02d" $bay][format "%02d" $story] $nodek2 $nodel1 \
                            $A $E $I_mod $pdelta
                }

                set hinge_to_hinge_dists($bay,$story) [expr {[nodeCoord $nodel1 1] - [nodeCoord $nodek2 1]}]
            } else {
                if {($num_bays % 2) && ($bay == $num_bays/2)} {
                    set nodem 2[format "%02d" $story]
                    element elasticBeamColumn 23[format "%02d" $bay][format "%02d" $story] $nodei $nodem $A \
                            $E $I_mod $pdelta
                    element elasticBeamColumn 23[format "%02d" [expr {$bay + 1}]][format "%02d" $story] \
                            $nodem $nodej $A $E $I_mod $pdelta
                } else {
                    element elasticBeamColumn 20[format "%02d" $bay][format "%02d" $story] $nodei $nodej \
                            $A $E $I_mod $pdelta
                }

                set hinge_to_hinge_dists($bay,$story) $beam_lengths($bay,$story)
            }
        }
    }

    # Create an elastic uniaxial material to be used for all truss elements
    uniaxialMaterial Elastic 200000 $E

    # Create the truss elements that restrain the basement levels from deflecting laterally
    if {$basement_flag} {
        for {set story 1} {$story <= $num_basement_levels} {incr story} {
            set beam_section [lindex [lindex $beam_shapes [expr {$story - 1}]] 0]
            set link_A [expr {100.0*[dict get $wshape_props $beam_section A]/[lindex $bay_widths 0]}]
            set nodei 400[format "%02d" $story]0
            set nodej 100[format "%02d" $story]3
            element truss 6000[format "%02d" $story] $nodei $nodej $link_A 200000

            set beam_section [lindex [lindex $beam_shapes [expr {$story - 1}]] end]
            set link_A [expr {100.0*[dict get $wshape_props $beam_section A]/[lindex $bay_widths end]}]
            set nodei 1[format "%02d" $num_bays][format "%02d" $story]3
            set nodej 4[format "%02d" $num_bays][format "%02d" $story]0
            element truss 60[format "%02d" $num_bays][format "%02d" $story] $nodei $nodej $link_A 200000
        }
    }
# puts "elements were designed"
    ########################################  Create Plastic Hinges  ########################################

    # Write the properties of all plastic hinges to a text file if the debug flag is true
    if {$debug_flag} {
        file mkdir Model_Info
        set hingefile [open Model_Info/plastic_hinge_info.out w]
    } else {
        set hingefile 0
    }
    
    # Loop over all bays
    for {set bay 0} {$bay <= $num_bays} {incr bay} {
        
        # Create a plastic hinge at the base of the lowermost column if there are no basement levels
        # Account for the fact that the column could contain a splice
        # Compute the shear span as half the member length (assuming double curvature response)
        if {!$basement_flag} {
            set cross_sections [split [lindex [lindex $col_shapes 0] $bay] ,]
            if {[llength $cross_sections] == 1} {
                set cross_section $cross_sections
            } else {
                set cross_section [lindex $cross_sections 0]
            }
            set cross_section_props [dict get $wshape_props $cross_section]
            set Lb $col_lengths($bay,1)
            set L [expr {$col_lengths($bay,1)/2.0}]
            set hinge_to_hinge_dist $col_lengths($bay,1)
            set P [lindex [lindex $col_axforce 0] $bay]
            set rot_spring_material [CreateRotationalSpring $bay 0 0 $cross_section_props $Lb $L \
                    $hinge_to_hinge_dist $Fy $E $n $P $debug_flag $hingefile false]

            # Identify nodei and nodej
            set nodei 1[format "%02d" $bay]004
            set nodej 1[format "%02d" $bay]002

            # Create the zero-length element
            element zeroLength 30[format "%02d" $bay]00 $nodei $nodej -mat $rot_spring_material -dir 6
            equalDOF $nodei $nodej 1 2
        }

        # Loop over all stories and create panel zone elements at each beam-column joint
        for {set story 1} {$story <= $num_stories} {incr story} {

            # Create uniaxial materials for the plastic hinges around the panel zone
            for {set node 1} {$node <= 4} {incr node} {
                
                # Choose the appropriate cross section to use based on the node location, compute its
                # properties and the unbraced length of the member in the direction of the plastic hinge
                # Compute the shear span as half the member length (assuming double curvature response) and
                # correct for the distance of the plastic hinge from the column face if RBS hinges or cover
                # plates are present
                switch $node {
                    1 {
                        if {$bay < $num_bays} {
                            set cross_section [lindex [lindex $beam_shapes [expr {$story - 1}]] $bay]
                            set cross_section_props [dict get $wshape_props $cross_section]
                            set d_beam_right [dict get $cross_section_props d]

                            switch $beam_Lb_flag {
                                1 {
                                    set Lb [lindex [lindex $beam_Lb [expr {$story - 1}]] $bay]
                                }
                                2 {
                                    set Lb $beam_lengths($bay,$story)
                                }
                                3 {
                                    set ry [dict get $cross_section_props ry]
                                    set Lb [expr {0.086*$ry*$E/$Fy}]
                                }
                                4 {
                                    set ry [dict get $cross_section_props ry]
                                    set Lb [expr {0.17*$ry*$E/$Fy}]
                                }
                            }

                            set L [expr {$beam_lengths($bay,$story)/2.0}]
                            if {$beam_rbs_flag || $beam_cover_plate_flag} {
                                set L [expr {$L - [lindex [lindex $beam_hinge_offsets [expr {$story - 1}]] \
                                        [expr {2*$bay}]]}]
                            }

                            set hinge_to_hinge_dist $hinge_to_hinge_dists($bay,$story)

                            set P 0.0
                            set rbs_flag $beam_rbs_flag
                            if {$rbs_flag} {
                                set rbs_width [lindex [lindex $rbs_widths [expr {$story - 1}]] \
                                        [expr {2*$bay}]]
                            } else {
                                set rbs_width 0.0
                            }
                        } else {
                            set cross_section {}
                            set d_beam_right 0.0
                        }
                    }
                    2 {
                        if {$story < $num_stories} {
                            set cross_sections [split [lindex [lindex $col_shapes $story] $bay] ,]
                            if {[llength $cross_sections] == 1} {
                                set cross_section $cross_sections
                            } else {
                                set cross_section [lindex $cross_sections 0]
                            }
                            set cross_section_props [dict get $wshape_props $cross_section]
                            set Lb $col_lengths($bay,[expr {$story + 1}])
                            set L [expr {$col_lengths($bay,[expr {$story + 1}])/2.0}]
                            set hinge_to_hinge_dist $col_lengths($bay,[expr {$story + 1}])
                            set P [lindex [lindex $col_axforce $story] $bay]
                            set rbs_flag false
                            set rbs_width 0.0
                        } else {
                            set cross_section {}
                        }
                    }
                    3 {
                        if {$bay > 0} {
                            set cross_section [lindex [lindex $beam_shapes [expr {$story - 1}]] \
                                    [expr {$bay - 1}]]
                            set cross_section_props [dict get $wshape_props $cross_section]
                            set d_beam_left [dict get $cross_section_props d]

                            switch $beam_Lb_flag {
                                1 {
                                    set Lb [lindex [lindex $beam_Lb [expr {$story - 1}]] [expr {$bay - 1}]]
                                }
                                2 {
                                    set Lb $beam_lengths([expr {$bay - 1}],$story)
                                }
                                3 {
                                    set ry [dict get $cross_section_props ry]
                                    set Lb [expr {0.086*$ry*$E/$Fy}]
                                }
                                4 {
                                    set ry [dict get $cross_section_props ry]
                                    set Lb [expr {0.17*$ry*$E/$Fy}]
                                }
                            }

                            set L [expr {$beam_lengths([expr {$bay - 1}],$story)/2.0}]
                            if {$beam_rbs_flag || $beam_cover_plate_flag} {
                                set L [expr {$L - [lindex [lindex $beam_hinge_offsets [expr {$story - 1}]] \
                                        [expr {2*$bay - 1}]]}]
                            }

                            set hinge_to_hinge_dist $hinge_to_hinge_dists([expr {$bay - 1}],$story)

                            set P 0.0
                            set rbs_flag $beam_rbs_flag
                            if {$rbs_flag} {
                                set rbs_width [lindex [lindex $rbs_widths [expr {$story - 1}]] \
                                        [expr {2*$bay - 1}]]
                            } else {
                                set rbs_width 0.0
                            }
                        } else {
                            set cross_section {}
                            set d_beam_left 0.0
                        }
                    }
                    4 {
                        set cross_sections [split [lindex [lindex $col_shapes [expr {$story - 1}]] $bay] ,]
                        if {[llength $cross_sections] == 1} {
                            set cross_section $cross_sections
                        } else {
                            set cross_section [lindex $cross_sections 1]
                        }
                        set cross_section_props [dict get $wshape_props $cross_section]
                        set Lb $col_lengths($bay,$story)
                        set L [expr {$col_lengths($bay,$story)/2.0}]
                        set hinge_to_hinge_dist $col_lengths($bay,$story)
                        set P [lindex [lindex $col_axforce [expr {$story - 1}]] $bay]
                        set rbs_flag false
                        set rbs_width 0.0
                        set panel_cross_section_props $cross_section_props
                    }
                }
                
                # Create the uniaxial material for the plastic hinge
                if {[llength $cross_section]} {
                    set rot_spring_materials($node) [CreateRotationalSpring $bay $story $node \
                            $cross_section_props $Lb $L $hinge_to_hinge_dist $Fy $E $n $P $debug_flag \
                            $hingefile $rbs_flag $rbs_width]
                } else {
                    set rot_spring_materials($node) 0
                }
            }

            # Compute the thickness of the doubler plate and the maximum depth of the two beams framing
            # into the panel zone
            set t_doubler [lindex [lindex $doubler_plate_thicknesses [expr {$story - 1}]] $bay]
            set d_beam [expr {max($d_beam_left, $d_beam_right)}]

            # Create the uniaxial material for the shear behavior of the panel zone
            set shear_spring_material [CreateShearSpring $bay $story $node $panel_cross_section_props $Fy \
                    $E $v $t_doubler $d_beam $debug_flag $hingefile]

            # Create the panel element and beam hinge elements if required
            if {$beam_rbs_flag || $beam_cover_plate_flag} {
                element Joint2D 40[format "%02d" $bay][format "%02d" $story] \
                                1[format "%02d" $bay][format "%02d" $story]1 \
                                1[format "%02d" $bay][format "%02d" $story]2 \
                                1[format "%02d" $bay][format "%02d" $story]3 \
                                1[format "%02d" $bay][format "%02d" $story]4 \
                                1[format "%02d" $bay][format "%02d" $story]0 \
                                0 $rot_spring_materials(2) \
                                0 $rot_spring_materials(4) \
                                $shear_spring_material 1

                if {$bay > 0} {
                    set nodei 1[format "%02d" $bay][format "%02d" $story]5
                    set nodej 1[format "%02d" $bay][format "%02d" $story]6

                    element zeroLength 31[format "%02d" $bay][format "%02d" $story] $nodei $nodej \
                            -mat $rot_spring_materials(3) -dir 6
                    equalDOF $nodei $nodej 1 2
                }
                if {$bay < $num_bays} {
                    set nodei 1[format "%02d" $bay][format "%02d" $story]7
                    set nodej 1[format "%02d" $bay][format "%02d" $story]8

                    element zeroLength 32[format "%02d" $bay][format "%02d" $story] $nodei $nodej \
                            -mat $rot_spring_materials(1) -dir 6
                    equalDOF $nodei $nodej 1 2
                }
            } else {
                element Joint2D 40[format "%02d" $bay][format "%02d" $story] \
                                1[format "%02d" $bay][format "%02d" $story]1 \
                                1[format "%02d" $bay][format "%02d" $story]2 \
                                1[format "%02d" $bay][format "%02d" $story]3 \
                                1[format "%02d" $bay][format "%02d" $story]4 \
                                1[format "%02d" $bay][format "%02d" $story]0 \
                                $rot_spring_materials(1) $rot_spring_materials(2) \
                                $rot_spring_materials(3) $rot_spring_materials(4) \
                                $shear_spring_material 1
            }
        }
    }

    if {$debug_flag} {
        close $hingefile
    }

    ########################################  Create Leaning Column  ########################################

    # Compute the location of the leaning column to the right of the frame
    set lc_bay [expr {$num_bays + 1}]
    set x_coord [expr {[lindex $x_coords end] + $bay_width}]

    # Create a node at the base of the leaning column and fix its x and y displacements only
    node 1[format "%02d" $lc_bay]002 $x_coord 0.0
    fix 1[format "%02d" $lc_bay]002 1 1 0

    # Create two nodes at the level of each story (except at the roof) and link their x and y displacements
    for {set story 1} {$story < $num_stories} {incr story} {
        set y_coord [lindex $y_coords $story]

        node 1[format "%02d" $lc_bay][format "%02d" $story]2 $x_coord $y_coord
        node 1[format "%02d" $lc_bay][format "%02d" $story]4 $x_coord $y_coord

        equalDOF 1[format "%02d" $lc_bay][format "%02d" $story]2 \
                 1[format "%02d" $lc_bay][format "%02d" $story]4 \
                 1 2
    }

    # Create a node at the roof level
    set y_coord [lindex $y_coords end]
    node 1[format "%02d" $lc_bay][format "%02d" $num_stories]4 $x_coord $y_coord

    # Set the moment of inertia of all leaning column elements to the largest among all the other elements
    # (This is almost inconsequential since the moments in the elements will be negligible)
    set I_strut $I_max

    # Create the struts of the leaning column
    for {set story 1} {$story <= $num_stories} {incr story} {

        # Set the area of the leaning column element by scaling the largest area among all the other elements
        # based on the fraction of the story load tributary to the gravity system
        set tributary_ratio_frame [lindex $tributary_ratios_frame [expr {$story - 1}]]
        set tributary_ratio_lc [expr {1.0 - $tributary_ratio_frame}]
        set A_strut [expr {$A_sum($story)*$tributary_ratio_lc/$tributary_ratio_frame}]

        # Identify nodei and nodej
        set nodei 1[format "%02d" $lc_bay][format "%02d" [expr {$story - 1}]]2
        set nodej 1[format "%02d" $lc_bay][format "%02d" $story]4

        element elasticBeamColumn 50[format "%02d" $lc_bay][format "%02d" $story] $nodei $nodej $A_strut \
                $E $I_strut $pdelta
    }

    # Link the middle of the frame to the leaning column using truss elements
    set A_strut [expr {100.0*$A_max}]

    for {set story 1} {$story <= $num_stories} {incr story} {

        # Identify nodei and nodej
        # If the number of bays is odd, connect the leaning column to the specially placed nodes in the
        # middle of the center bay, else, connect it to the right of the joint panel in the center column
        if {$num_bays % 2} {
            set nodei 2[format "%02d" $story]
        } else {
            set nodei 1[format "%02d" [expr {$num_bays/2}]][format "%02d" $story]1
        }
        set nodej 1[format "%02d" $lc_bay][format "%02d" $story]4

        element truss 51[format "%02d" $lc_bay][format "%02d" $story] $nodei $nodej $A_strut 200000
    }

    #########################################  Assign Nodal Masses  #########################################

    # Loop over all stories to assign masses to the nodes in the frame and leaning column
    for {set story 1} {$story <= $num_stories} {incr story} {
        
        # Compute the total mass at the story and the fractions tributary to the frame and leaning column
        set story_mass [lindex $story_masses [expr {$story - 1}]]
        set tributary_ratio_frame [lindex $tributary_ratios_frame [expr {$story - 1}]]
        set story_mass_frame [expr {$tributary_ratio_frame*$story_mass}]
        set story_mass_lc [expr {$story_mass - $story_mass_frame}]

        # Compute the linear mass density at the story
        set story_mass_density [expr {$story_mass_frame/[lindex $x_coords end]}]

        # Declare a variable to keep track of the sum of the moments of inertia at all the beam-column joints
        # in the story. This will later be used to compute the moment of inertia to apply to the leaning
        # column nodes.
        set mom_sum_story 0.0

        # Loop over all beam-column joints in the story
        for {set bay 0} {$bay <= $num_bays} {incr bay} {

            # Compute the mass at the beam-column joint based on the fraction of the area tributary to the
            # frame that is tributary to the column
            set tributary_ratio_column [lindex $tributary_ratios_columns $bay]
            set joint_mass($bay,$story) [expr {$story_mass_frame*$tributary_ratio_column}]

            # Compute the number of nodes in the beam-column joint and then the mass to be assigned to each
            # node
            if {$beam_rbs_flag || $beam_cover_plate_flag} {
                if {$bay == 0 || $bay == $num_bays} {
                    set num_nodes_joint 6
                } else {
                    set num_nodes_joint 8
                }
            } else {
                set num_nodes_joint 4
            }

            set node_mass [expr {$joint_mass($bay,$story)/$num_nodes_joint}]

            # Compute the moment of inertia at the beam-column joint depending on whether it is an exterior
            # or interior joint
            # Assume that the load up to a distance equal to the depth of the beam from either face of the
            # column contributes to the moment of inertia at the joint
            set col_sections [split [lindex [lindex $col_shapes [expr {$story - 1}]] $bay] ,]
            if {[llength $col_sections] == 1} {
                set col_section $col_sections
            } else {
                set col_section [lindex $col_sections 1]
            }
            set col_depth [dict get $wshape_props $col_section d]

            if {$bay > 0} {
                set beam_section [lindex [lindex $beam_shapes [expr {$story - 1}]] [expr {$bay - 1}]]
                set beam_depth_left [dict get $wshape_props $beam_section d]
            } else {
                set beam_depth_left 0.0
            }

            if {$bay < $num_bays} {
                set beam_section [lindex [lindex $beam_shapes [expr {$story - 1}]] $bay]
                set beam_depth_right [dict get $wshape_props $beam_section d]
            } else {
                set beam_depth_right 0.0
            }

            set contr_length_left [expr {$col_depth/2.0 + 1.01*$beam_depth_left}]
            set contr_length_right [expr {$col_depth/2.0 + 1.01*$beam_depth_right}]
            set joint_mom($bay,$story) [expr {$story_mass_density*($contr_length_left**3 + \
                    $contr_length_right**3)/3.0}]

            # Compute the moment of inertia to be assigned to each node in the beam-column joint
            set node_mom [expr {$joint_mom($bay,$story)/$num_nodes_joint}]

            # Loop over all nodes in the beam-column joint and set their masses and moments of inertia
            for {set node 1} {$node <= 4} {incr node} {
                mass 1[format "%02d" $bay][format "%02d" $story]$node $node_mass $node_mass $node_mom
            }
            
            if {$beam_rbs_flag || $beam_cover_plate_flag} {
                if {$bay == 0} {
                    set hinge_nodes {7 8}
                } elseif {$bay == $num_bays} {
                    set hinge_nodes {5 6}
                } else {
                    set hinge_nodes {5 6 7 8}
                }

                foreach node $hinge_nodes {
                    mass 1[format "%02d" $bay][format "%02d" $story]$node $node_mass $node_mass $node_mom
                }
            }

            # If a splice exists in the column below the beam-column joint, assign mass and moment of inertia
            # to it
            if {[llength $col_sections] != 1} {
                set col_section [lindex $col_sections 0]
                set col_weight [dict get $wshape_props $col_section W]
                set col_depth [dict get $wshape_props $col_section d]

                set node_mass [expr {$col_weight*$lbf/$ft*$col_lengths($bay,$story)/$g}]
                set node_mom [expr {$col_weight*$lbf/$ft*2*($col_depth**3)/3.0}]

                mass 3[format "%02d" $bay][format "%02d" $story] $node_mass $node_mass $node_mom
            }

            # Update the sum of the moments of inertia at all beam-column joints in the story
            set mom_sum_story [expr {$mom_sum_story + $joint_mom($bay,$story)}]
        }

        # If a node has been defined in the center of the middle bay to connect the leaning column,
        # assign mass and moment of inertia to it
        if {$num_bays % 2} {
            set beam_section [lindex [lindex $beam_shapes [expr {$story - 1}]] [expr {$num_bays/2}]]
            set beam_weight [dict get $wshape_props $beam_section W]
            set beam_depth [dict get $wshape_props $beam_section d]

            set node_mass [expr {$beam_weight*$lbf/$ft*$beam_lengths([expr {$num_bays/2}],$story)/$g}]
            set node_mom [expr {$beam_weight*$lbf/$ft*2*($beam_depth**3)/3.0}]

            mass 2[format "%02d" $story] $node_mass $node_mass $node_mom
        }

        # Assign mass and moment of inertia to the nodes in the leaning column
        set joint_mass($lc_bay,$story) $story_mass_lc
        set joint_mom($lc_bay,$story) [expr {$mom_sum_story*$story_mass_lc/$story_mass_frame}]

        if {$story < $num_stories} {
            set node_mass [expr {$joint_mass($lc_bay,$story)/2.0}]
            set node_mom [expr {$joint_mom($lc_bay,$story)/2.0}]

            mass 1[format "%02d" $lc_bay][format "%02d" $story]2 $node_mass $node_mass $node_mom
            mass 1[format "%02d" $lc_bay][format "%02d" $story]4 $node_mass $node_mass $node_mom
        } else {
            set node_mass $joint_mass($lc_bay,$story)
            set node_mom $joint_mom($lc_bay,$story)

            mass 1[format "%02d" $lc_bay][format "%02d" $story]4 $node_mass $node_mass $node_mom 
        }
    }

    # Assign a moment of inertia to all the nodes at ground level
    set mom_sum_ground 0.0
    for {set bay 0} {$bay <= $num_bays} {incr bay} {
        set col_sections [split [lindex [lindex $col_shapes 0] $bay] ,]
        if {[llength $col_sections] == 1} {
            set col_section $col_sections
        } else {
            set col_section [lindex $col_sections 0]
        }
        set col_weight [dict get $wshape_props $col_section W]
        set col_depth [dict get $wshape_props $col_section d]

        set joint_mom($bay,0) [expr {$col_weight*$lbf/$ft*($col_depth**3)/3.0}]
        set mom_sum_ground [expr {$mom_sum_ground + $joint_mom($bay,0)}]

        mass 1[format "%02d" $bay]002 0.0 0.0 $joint_mom($bay,0)
    }
    set joint_mom($lc_bay,0) [expr {$mom_sum_ground*$story_mass_lc/$story_mass_frame}]
    mass 1[format "%02d" $lc_bay]002 0.0 0.0 $joint_mom($lc_bay,0)

    #########################################  Apply Gravity Load  #########################################

    # Define the gravity load pattern
    set gravity_load_ts 1
    timeSeries Linear $gravity_load_ts

    set gravity_load_pattern 1
    pattern Plain $gravity_load_pattern $gravity_load_ts {
        
        # Loop over all bays and stories to apply gravity loads at each beam-column joint and the leaning
        # column
        for {set story 1} {$story <= $num_stories} {incr story} {
            
            # Compute the story load tributary to the frame and leaning column if the loads are specified
            # separate from the seismic mass (i.e. if story_load_flag is set to 2)
            if {$story_load_flag == 2} {
                set story_load [lindex $story_loads [expr {$story - 1}]]
                set tributary_ratio_frame [lindex $tributary_ratios_frame [expr {$story - 1}]]
                set story_load_frame [expr {$tributary_ratio_frame*$story_load}]
                set story_load_lc [expr {$story_load - $story_load_frame}]
            }

            # Apply a gravity load at every beam-column joint in the story
            for {set bay 0} {$bay <= $num_bays} {incr bay} {
                switch $story_load_flag {
                    1 {
                        # Compute the load at the beam-column joint from the previously computed seismic
                        # masses at each beam-column joint
                        set joint_load($bay,$story) [expr {-$joint_mass($bay,$story)*$g}]
                    }
                    2 {
                        # Compute the load at the beam-column joint based on the story load tributary to the
                        # frame and the fraction of this area that is tributary to the column
                        set tributary_ratio_column [lindex $tributary_ratios_columns $bay]
                        set joint_load($bay,$story) [expr {-$story_load_frame*$tributary_ratio_column}]
                    }
                }

                load 1[format "%02d" $bay][format "%02d" $story]2 \
                        0.0 $joint_load($bay,$story) 0.0
            }

            # Compute and apply a gravity load at the leaning column
            switch $story_load_flag {
                1 {
                    # Compute the load at the leaning column from the previously computed seismic mass
                    set joint_load($lc_bay,$story) [expr {-$joint_mass($lc_bay,$story)*$g}]
                }
                2 {
                    # Use the previously computed story load tributary to the leaning column
                    set joint_load($lc_bay,$story) -$story_load_lc
                }
            }

            if {$story < $num_stories} {
                load 1[format "%02d" $lc_bay][format "%02d" $story]2 \
                        0.0 $joint_load($lc_bay,$story) 0.0
            } else {
                load 1[format "%02d" $lc_bay][format "%02d" $story]4 \
                        0.0 $joint_load($lc_bay,$story) 0.0
            }
        }
    }

    # source [file join [file dirname [info script]] original/display2D.tcl]		
	
    # Apply the gravity load in 10 steps and hold it constant for the rest of the analysis
    set numsteps 10
    set tol 1e-12
    set maxiter 20

    constraints Transformation
    numberer RCM
    system SparseGEN
    test RelativeEnergyIncr $tol $maxiter
    algorithm Newton
    integrator LoadControl [expr {1.0/$numsteps}]
    analysis Static

    if {[analyze $numsteps]} {
        puts "Application of gravity load failed"
    }

    puts "Gravity is done!"	
    loadConst -time 0.0
    wipeAnalysis

    ###########################################  Define Damping  ###########################################

    # Set percentage of critical damping and two mode numbers to apply it to. The frequencies of these modes,
    # scaled by the provided factors, will be used to set the Rayleigh damping coefficients.
    set damping 0.02
    set modes {1 3}
    set freq_scales {0.8 1.0}

    set eigenvalues [eigen -genBandArpack [lindex $modes 1]]
    set omega1 [expr {[lindex $freq_scales 0]*sqrt([lindex $eigenvalues [lindex $modes 0]-1])}]
    set omega2 [expr {[lindex $freq_scales 1]*sqrt([lindex $eigenvalues [lindex $modes 1]-1])}]
    set alphaM [expr {2*$damping*$omega1*$omega2/($omega1 + $omega2)}];
    set betaK [expr {2*$damping/($omega1 + $omega2)}];

    # Apply damping to only the linear elastic frame elements, excluding all plastic hinges and the leaning
    # column. Use the committed stiffness matrix at each step to form the damping matrix.
    region 12 -eleRange 100000 299999 -rayleigh $alphaM 0.0 0.0 $betaK
}

# Create a Modified Ibarra-Medina-Krawinkler bilinear material that models the relation between moment and
# rotation in a steel moment frame plastic hinge
# Return the integer tag of the material created
proc CreateRotationalSpring {bay story node cross_section_props Lb L hinge_to_hinge_dist Fy E n P \
        debug_flag hingefile rbs_flag {rbs_width 0.0}} {

    dict with cross_section_props {
        
        # Compute Mc_My, thetap, thetapc, Lambda, and kappa from the ATC-72 equations
        set hinge_props [HingeProperties $d $bf_2tf $h_tw $ry $Lb $L $Fy $rbs_flag]

        dict with hinge_props {

            # Compute the rotational spring properties
            set K [expr {6.0*$E*$Ix/$hinge_to_hinge_dist}]
            set K_mod [expr {($n + 1.0)*$K}]

            # If an RBS cut is present, modify Zx accordingly
            if {$rbs_flag} {
                set Zx [expr {$Zx - ($bf - $rbs_width)*$tf*($d - $tf)}]
            }
            set My [expr {$My_Myp*$Zx*$Fy}]

            # If the member has an axial load, compute the yield moment using an interaction curve
            # Also modify thetap and thetapc if rx/ry > 3 (criterion formulated based on recommendations from
            # ATC-72)
            if {$P} {
                set Py [expr {$A*$Fy}]
                set p [expr {$P/$Py}]
                set m [expr {sqrt((1.0 - ($p**2))/(1.0 + 3.5*($p**2)))}]
                set My [expr {$m*$My}]

                if {[expr {$rx/$ry}] > 3.0 && $p > 0.2} {
                    set reduction_factor [expr {1.0 - sqrt((($p - 0.2)/0.8)**2)}]
                    set thetap [expr {$reduction_factor*$thetap}]
                    set thetapc [expr {$reduction_factor*$thetapc}]
                }
            }

            set thetay_mod [expr {$My/$K_mod}]

            set alphas [expr {$My*($Mc_My - 1.0)/($thetap*$K)}]
            set alphas_mod [expr {$alphas/(1.0 + $n*(1.0 - $alphas))}]
            set thetap_mod [expr {$My*($Mc_My - 1.0)/($alphas_mod*$K_mod)}]

            set alphapc [expr {-$My*$Mc_My/($thetapc*$K)}]
            set alphapc_mod [expr {$alphapc/(1.0 + $n*(1.0 - $alphapc))}]
            set thetapc_mod [expr {-$My*$Mc_My/($alphapc_mod*$K_mod)}]

            set thetau_mod 0.20;
            set c 1.0
            set D 1.0

            # Create the rotational spring material
            set rot_spring_material 1[format "%02d" $bay][format "%02d" $story]$node
            uniaxialMaterial Bilin $rot_spring_material $K_mod $alphas_mod $alphas_mod $My -$My $Lambda \
                    $Lambda $Lambda $Lambda $c $c $c $c $thetap_mod $thetap_mod $thetapc_mod $thetapc_mod \
                    $kappa $kappa $thetau_mod $thetau_mod $D $D

            # Write information about the plastic hinge to the plastic hinge text file if the debug flag is
            # set to true
            if {$debug_flag} {
                puts $hingefile "Hinge [format "%02d" $bay][format "%02d" $story]$node"
                puts $hingefile "My: $My"
                puts $hingefile "Mc: [expr {$My*$Mc_My}]"
                puts $hingefile "kappa: $kappa"
                puts $hingefile "thetay: $thetay_mod"
                puts $hingefile "thetac: [expr {$thetay_mod + $thetap_mod}]"
                puts $hingefile "thetau: $thetau_mod"
                puts $hingefile "K: $K_mod"
                puts $hingefile "alphas: $alphas_mod"
                puts $hingefile "alphapc: $alphapc_mod"
                puts $hingefile "Lambda: $Lambda"
                puts $hingefile ""
            }
        }
    }

    return $rot_spring_material
}

# Create a Hysteretic material that models the relation between shear-equivalent moment and shear stress
# in a steel moment frame panel zone
# Return the integer tag of the material created
proc CreateShearSpring {bay story node cross_section_props Fy E v t_doubler d_beam debug_flag hingefile} {

    dict with cross_section_props {

        # Compute the shear spring properties
        set G [expr {$E/(2.0*(1.0 + $v))}]

        set Ke [expr {0.95*$G*$d*($tw + $t_doubler)}]
        set gammay [expr {$Fy/(sqrt(3.0)*$G)}]
        set Vy [expr {$Ke*$gammay}]
        set My [expr {$Vy*$d_beam}]

        set Kp [expr {0.95*$G*$bf*($tf**2)/$d_beam}]
        set gammap [expr {4.0*$gammay}]
        set Vp [expr {$Vy + $Kp*($gammap - $gammay)}]
        set Mp [expr {$Vp*$d_beam}]

        # 1% strain hardening is used despite the recommended 3% since Kp is approximately equal to 3% of Ke
        # and K_final has to be lower than Kp
        set K_final [expr {0.01*$Ke}]
        set gamma_final [expr {100.0*$gammay}]
        set V_final [expr {$Vp + $K_final*($gamma_final - $gammap)}]
        set M_final [expr {$V_final*$d_beam}]
    }

    # Create the shear spring material
    set shear_spring_material 1[format "%02d" $bay][format "%02d" $story]0
    uniaxialMaterial Hysteretic $shear_spring_material $My $gammay $Mp $gammap $M_final $gamma_final \
            -$My -$gammay -$Mp -$gammap -$M_final -$gamma_final 1.0 1.0 0.0 0.0 0.0

    # Write information about the plastic hinge to the plastic hinge text file if the debug flag is set to
    # true
    if {$debug_flag} {
        puts $hingefile "Hinge [format "%02d" $bay][format "%02d" $story]0"
        puts $hingefile "Vy: $Vy"
        puts $hingefile "Vp: $Vp"
        puts $hingefile "My: $My"
        puts $hingefile "Mp: $Mp"
        puts $hingefile "gammay: $gammay"
        puts $hingefile "gammap: $gammap"
        puts $hingefile "Ke: $Ke"
        puts $hingefile "Kp: $Kp"
        puts $hingefile "K_final: $K_final"
        puts $hingefile ""
    }

    return $shear_spring_material
}

