##############################################################################################################
# Reagan Chandramohan                                                                                        #
# John A. Blume Earthquake Engineering Center                                                                #
# Stanford University                                                                                        #
# Last edited: 27-May-2014                                                                                   #
##############################################################################################################

# Compute the properties of a plastic hinge in a steel W-shape section using empirical equations based on
# Lignos, D. G., and Krawinkler, H. (2011). “Deterioration Modeling of Steel Components in Support of
# Collapse Prediction of Steel Moment Frames under Earthquake Loading.” Journal of Structural Engineering,
# 137(11), 1291–1302.
# 
# They are slightly different from the ones available in:
# PEER. (2010). Modeling and acceptance criteria for seismic design and analysis of tall buildings.
# Berkeley, CA: PEER/ATC-72-1, Pacific Earthquake Engineering Research Center.
# 
# For non-RBS connections, the equations are valid for 4 in <= d <= 36 in
# For RBS connections, the equations are valid for 18 in <= d <= 36 in

proc HingeProperties {d bf_2tf h_tw ry Lb L Fy rbs_flag} {

    # Common terms
    set Lb_ry [expr {$Lb/$ry}]
    set L_d [expr {$L/$d}]
    set d_21 [expr {$d/21.0}]
    set Fy_50 [expr {$Fy/50.0}]

    # Use different equations depending on whether the plastic hinge is an RBS cut
    if {$rbs_flag} {

        # thetap
        set terms "$h_tw $bf_2tf $Lb_ry $L_d $d_21 $Fy_50"
        set coeffs {-0.314 -0.100 -0.185 0.113 -0.760 -0.070}
        set thetap 0.19
        foreach term $terms coeff $coeffs {
            set thetap [expr {$thetap * ($term ** $coeff)}]
        }
        dict set hinge_props thetap $thetap

        # thetapc
        set terms "$h_tw $bf_2tf $Lb_ry $Fy_50"
        set coeffs {-0.513 -0.863 -0.108 -0.360}
        set thetapc 9.52
        foreach term $terms coeff $coeffs {
            set thetapc [expr {$thetapc * ($term ** $coeff)}]
        }
        dict set hinge_props thetapc $thetapc

        # Lambda
        set terms "$h_tw $bf_2tf $Lb_ry $Fy_50"
        set coeffs {-1.140 -0.632 -0.205 -0.391}
        set Lambda 585.0
        foreach term $terms coeff $coeffs {
            set Lambda [expr {$Lambda * ($term ** $coeff)}]
        }
        dict set hinge_props Lambda $Lambda

        # My over Myp
        dict set hinge_props My_Myp 1.06

        # Mc over My
        dict set hinge_props Mc_My 1.09

    } else {

        # Use different equations depending on the depth of the cross section
        if {$d < 21.0} {

            # thetap
            set terms "$h_tw $bf_2tf $L_d $d_21 $Fy_50"
            set coeffs {-0.365 -0.140 0.340 -0.721 -0.230}
            set thetap 0.0865
            foreach term $terms coeff $coeffs {
                set thetap [expr {$thetap * ($term ** $coeff)}]
            }
            dict set hinge_props thetap $thetap

            # thetapc
            set terms "$h_tw $bf_2tf $d_21 $Fy_50"
            set coeffs {-0.565 -0.800 -0.280 -0.430}
            set thetapc 5.63
            foreach term $terms coeff $coeffs {
                set thetapc [expr {$thetapc * ($term ** $coeff)}]
            }
            dict set hinge_props thetapc $thetapc

            # Lambda
            set terms "$h_tw $bf_2tf $Fy_50"
            set coeffs {-1.34 -0.595 -0.360}
            set Lambda 495.0
            foreach term $terms coeff $coeffs {
                set Lambda [expr {$Lambda * ($term ** $coeff)}]
            }
            dict set hinge_props Lambda $Lambda

        } else {
            
            # thetap
            set terms "$h_tw $bf_2tf $Lb_ry $L_d $d_21 $Fy_50"
            set coeffs {-0.550 -0.345 -0.023 0.090 -0.330 -0.130}
            set thetap 0.318
            foreach term $terms coeff $coeffs {
                set thetap [expr {$thetap * ($term ** $coeff)}]
            }
            dict set hinge_props thetap $thetap

            # thetapc
            set terms "$h_tw $bf_2tf $Lb_ry $d_21 $Fy_50"
            set coeffs {-0.610 -0.710 -0.110 -0.161 -0.320}
            set thetapc 7.50
            foreach term $terms coeff $coeffs {
                set thetapc [expr {$thetapc * ($term ** $coeff)}]
            }
            dict set hinge_props thetapc $thetapc

            # Lambda
            set terms "$h_tw $bf_2tf $Lb_ry $Fy_50"
            set coeffs {-1.26 -0.525 -0.130 -0.291}
            set Lambda 536.0
            foreach term $terms coeff $coeffs {
                set Lambda [expr {$Lambda * ($term ** $coeff)}]
            }
            dict set hinge_props Lambda $Lambda
        }

        # My over Myp
        dict set hinge_props My_Myp 1.17

        # Mc over My
        dict set hinge_props Mc_My 1.11
    }

    # Residual strength ratio (residual strength / yield strength)
    dict set hinge_props kappa 0.4

    return $hinge_props
}
