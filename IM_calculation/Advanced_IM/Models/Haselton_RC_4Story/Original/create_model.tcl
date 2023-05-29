##############################################################################################################
# Reagan Chandramohan                                                                                        #
# John A. Blume Earthquake Engineering Center                                                                #
# Stanford University                                                                                        #
# Last edited: 05-Oct-2014                                                                                   #
##############################################################################################################

# Create a modified version of Haselton's RC moment frame model from the directory 'modeldir'

##############################################################################################################

# Declare a 2d model
model BasicBuilder -ndm 2

# Define constants
set pi [expr {2.0*asin(1.0)}]
set g 386.089

# Source a subset of Haselton's procedures

# set modelname [glob -directory [file join [file dirname [info script]]] -type d *]
set modelname "ID1008_4Story"
puts $modelname

source [file join [file dirname [info script]] haselton_procedures.tcl]

# Define Clough model parameters
set c 1.0
set resStrRatio 0.01

# Define factors to distribute the element's stiffness among the beam-column element and the plastic hinges
set stiffFactor1 11.0
set stiffFactor2 1.1

# Define parameters for beams and columns (average values used for all beams and columns)
set ABm 1025.0
set Acol 1320.0
set EConcr 4030.0
set EAcol [expr {$EConcr*$Acol}]
set EABm [expr {$EConcr*$ABm}]
set EOfUnity 1.0

# Define geometric transformations
set primaryGeomTransT 1
geomTransf LinearWithPDelta $primaryGeomTransT

# Define joint panel parameters
set lrgDsp 1
set E_elasticTestMaterial [expr {1700.0*29000.0}]
uniaxialMaterial Elastic 49999 $E_elasticTestMaterial

# Define leaning column parameters
set A_strut [expr {100.0*$Acol}]
set E_strut $EOfUnity
set I_strut 1e9
set strutMatT 599
uniaxialMaterial Elastic $strutMatT $E_strut

# Define the small masses used for the three degrees of freedom
set smallMass1 2e-3
set smallMass2 2e-3
set smallMass3 4.1e-1
#set smallMass1 0.03235
#set smallMass2 0.03235
#set smallMass3 4.313

# Define the damping parameters
set dampRat 0.05
set dampRatF 1.0
set modes {1 3}

# Build the model using the parameters defined above
source  [file join [file dirname [info script]] $modelname/build_modified_haselton_model.tcl]
