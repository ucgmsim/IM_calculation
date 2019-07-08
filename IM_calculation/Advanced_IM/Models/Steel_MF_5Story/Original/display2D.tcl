# Display Model

# recorder display $windowTitle $xLoc $yLoc $xPixels $yPixels <-file $fileName>
# prp $X $Y $Z
# vup $Xv $Yv $Zv
# vpn $Xn $Yn $Zn
# viewWindow $Xprp,n $Xprp,p $Yprp,n $Yprp,p
# display $arg1 $arg2 $arg3


set h 200
recorder display "2D Shape" 10 10 500 500 
prp $h $h 1
vup 0 1 0
vpn 0 0 1
viewWindow -800 800 -800 800
display 1 5 20
after 10000
# while {1} {}