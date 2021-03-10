# Display Model

# recorder display $windowTitle $xLoc $yLoc $xPixels $yPixels <-file $fileName>   # position of window and resulotion 
# prp $X $Y $Z                                                                    # eye position 
# vup $Xv $Yv $Zv
# vpn $Xn $Yn $Zn
# viewWindow $Xprp,n $Xprp,p $Yprp,n $Yprp,p                                   # Controls Zoom of the display 
# display $arg1 $arg2 $arg3                                                    # arg1: mode shapes (Modes:-1,-2,...),(disp.: 1,2,...) 
                                                                                                                                                              # arg2: Node magnification  
                                                                                                                                                              # arg3: magnification


set h 200
recorder display "2D Shape" 10 10 500 500 
prp $h $h 1
vup 0 1 0
vpn 0 0 1
viewWindow -1200 1200 -1200 1200
display 1 5 20
after 10000
# while {1} {}