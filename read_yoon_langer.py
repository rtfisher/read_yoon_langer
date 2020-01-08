#######################################################################
## 
## Read Yoon and Langer (2005) plotfiles. Models are computed via the
##  self-consistent field method according to a specific angular 
##  momentum accretion model. Models are stored on a 2D spherical
##  mesh in mu = cos theta and spherical radius r, nth x nr in extent.
##
## Furthermore, the model density, angular velocity, and potentials
##  are stored in a dimensionless fashion. The dimensionless variables
##  (here noted as primed variables) are defined as 
##
##     r ' = r / rmax
##     rho' = rho / rhomax
##     w'   = w / sqrt (G * rhomax)
##     psi' = psi / (G * rmax**2. * rhomax)
##     phi' = phi / (G * rmax**2. * rhomax)
##
## We rescale all quantities to obtain physical variables after readin.
##
## The file format was described in a private e-mail communication
##  from Yoon, see yoon_langer_2005_format.txt file.
##
## Full details of their models are described in their 2005 A&A paper,
##  doi:10.1051/0004-6361:20042542.
##
#######################################################################

import math
import glob
import numpy as np
import sys

# Define mathematical and physical constants

pi = math.pi
G = 6.674e-8 # CGS units
msun = 1.987e33 # CGS
smallfloat = sys.float_info.min # Store a minimum sized float

# Define pressure according to completely degenerate electron gas,
#  from Ostriker & Bodenheimer (1968), used by Yoon & Langer (2005).
#  Take as arguments the dimensionless density rho as well as the central
#  density rhomax, return pressure in CGS (dyne / cm^2)

def pressure (rho, rhomax):

  mue = 2.    # assume Y_e = 0.5 composition (e.g. pure C/O)
  A = 6.01e22 # dyne / cm^2 
  B = 9.82e5 * mue # g / cm^3

  rho = rho * rhomax
  x = (rho / B)**(1./3.)
  p = A * (x * (2 * x**2. - 3.) * (x**2. + 1)**(1./2.) + 3. * math.asinh (x) )

  return p

files = glob.glob ("plot*")
files.sort() # process files in order

for filename in files :
  print ("Reading in file ", filename)
  file = open (filename, "r")

# First, let's read in the header information

  line = file.readline()  # first line
  lst  = line.split ()    # split this line into a string list lst
    
  nth = int (lst [0])
  nr = int (lst [1])
  nmd = int (lst [2])

  line = file.readline()  # second line
  line = line.replace ("D", "E") # replace "D" to "E" to process as floats
  lst = line.split()      # split the second line into a string list lst

  rmax = float (lst [0]) # equatorial radius
  mtot = float (lst [1]) # total mass
  rhomax = float (lst [2]) # maximum density
  tw = float (lst [3]) # T / W ratio of rotational to gravitational energy
  jtot = float (lst [4]) # total angular momentum
  egrav = float (lst [5]) # gravitational energy W
  erot = float (lst [6]) # rotational energy
  pi3 = float (lst [7]) # volume integration of pressure times 3 (virial term)
  temp = float (lst [8]) # ignored
  frhocrit = float (lst [9]) # f_p defined in paper
  w0 = float (lst [10]) # central angular velocity
  ws = float (lst [11]) # surface angular velocity
  bfit = float (lst [12]) # parameter a defined in paper

  print ("  rhomax = ", rhomax)
  print ("  mtot = ", mtot / msun, " solar masses")
  print ("  T / W = ", tw)

# Next, let's read in 2D model on the (cos theta) x (spherical radius) grid

  counter = 0 # line counter beginning at 0
  totalmass = 0. # initialize totalmass check
  totalJ    = 0. # total angular momentum check
  totalEg   = 0. # total gravitational energy check
  totalErot = 0. # total rotational energy check
  totalpi3  = 0. # 3 P V virial term

# Declare space for the 2D array model data on nth x nr mesh

  rhoarr = np.ndarray (shape = (nth, nr), dtype = float)
  warr   = np.ndarray (shape = (nth, nr), dtype = float)
  psiarr = np.ndarray (shape = (nth, nr), dtype = float)
  phiarr = np.ndarray (shape = (nth, nr), dtype = float)

# In following, note we are using Python indexing from 0 to N - 1,
#   which translates from Sung-Chul's Fortran indexing from 1 to N.

  mu     = np.arange (nth) / (nth - 1.)
  r      = 16. * np.arange (nr) / (nr - 1.) / 15.

  for line in file:
    line = line.replace ("D", "E") # replace "D" to "E" to process as floats
    lst = line.split()

    rho = float (lst [0]) # dimensionless density
    w   = float (lst [1]) # dimensionless angular velocity
    psi = float (lst [2]) # dimensionless total potential
    phi = float (lst [3]) # dimensionless gravitational potential

    j = counter % nr           # set innermost array index in r
    i = (counter - j) // nth   # set outer array index in (cos theta)

    rhoarr [i, j] = rho
    warr   [i, j] = w
    psiarr [i, j] = psi
    phiarr [i, j] = phi

    sinth  = (1. - (mu [i])**2.)**(1./2.)     # sintheta 
    z      = mu [i] * r [j]                   # z coordinate = r cos theta
    rcyl   = r [j] * sinth                    # cylindrical radius rcyl
                                              #   = r sin theta

# Define spherical volume element 4 pi r^2 dr d (cos theta), taking into
#  account only upper plane is modeled (mu >= 0), so incorporate an additional
#  factor of 2 for the symmetric lower half plane (mu < 0). 

    if (j != 0) :
      dr = r [j] - r [j - 1]
    else :
      dr = r [j + 1] - r [j]

    if (i != 0) :
      dcosth = mu [i] - mu [i - 1]
    else :
      dcosth = mu [i + 1] - mu [i]

    dV = 4. * pi * (r [j])**2. * dcosth * dr

    dM = rho * dV               # mass in cell
    vphi = w * rcyl             # angular velocity
    dJ = dM * rcyl * vphi       # angular momentum in cell   
    dEg= dM * phi               # gravitational energy 
    dErot = 0.5 * dM * vphi**2. # rotational energy
    pdV = pressure (rho, rhomax) * dV # p dV used in virial computation
     
# Compute global quantities as cross-checks
    totalmass += dM
    totalJ    += dJ
    totalEg   += dEg
    totalErot += dErot
    totalpi3  += pdV
    counter   += 1
    #end loop over line in file

# Lastly, let's scale dimensional quantities, using scalings for
#  mass, length, and time

  massfac   = rhomax * rmax**3.
  lengthfac = rmax
  timefac   = (G * rhomax)**(-1./2.)

  totalmass = totalmass * massfac  
  totalJ    = totalJ    * massfac * lengthfac**2. / timefac
  totalEg   = totalEg   * massfac * lengthfac**2. / timefac**2.
  totalErot = totalErot * massfac * lengthfac**2. / timefac**2.
  totalpi3  = totalpi3  * lengthfac**3. # note pressure is dimensional
  
  print ("   Total computed mass = ", totalmass / msun, " solar masses") 
  print ("   Fractional mass error = ", (totalmass - mtot) / mtot)

# Add small float to denominator for zero J cases; similarly below for erot
  print ("   Total computed J = ", totalJ)
  print ("   Fractional J error = ", (totalJ - jtot) / (jtot + smallfloat))

  print ("   Total computed Eg = ", totalEg)
  print ("   Fractional Eg error = ", (totalEg - egrav) / egrav)

  print ("   Total computed Erot = ", totalErot)
  print ("   Fractional Erot error = ", (totalErot - erot) / (erot + smallfloat)) 

  print ("   Total computed 3 P V = ", totalpi3)
  print ("   Fractional 3 P V error = ", (totalpi3 - pi3) / pi3 )

  print ("   Virial error = ", abs (2 * erot - egrav + 3. * pi3) / egrav)
  print ("   Computed virial error = ", abs (2. * totalErot - totalEg + 3 * totalpi3) / totalEg)

  r = r * lengthfac
  rho = rho * (massfac / lengthfac**3.) 
  w = w / timefac
  psi = psi * lengthfac**2. / timefac**2. # potential energy = energy / mass
  phi = phi * lengthfac**2. / timefac**2.
  #end loop over files
