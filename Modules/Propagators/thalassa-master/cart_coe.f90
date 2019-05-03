module CART_COE
! Description:
!    Routines to convert between Cartesian coordinates in the inertial frame
!    and classical orbital elements.
!
! Author:
!    Davide Amato
!    Space Dynamics Group - Technical University of Madrid
!    davideamato@email.arizona.edu
!
! ==============================================================================

use KINDS,      only: dk
use PHYS_CONST, only: pi,twopi,mzero
implicit none

contains

subroutine COE2CART(COE,R,V,GM,tolKE)
! Description:
!    Computes Cartesian state (R,V) from classical orbital elements COE.

! MODULES
use KEPLER, only: KESOLVE

! VARIABLES
implicit none
! Arguments
real(dk),intent(in)           ::  COE(1:6),GM
real(dk),intent(in),optional  ::  tolKE
real(dk),intent(out)          ::  R(1:3),V(1:3)
!logical,intent(out)           ::  KEflag
! Locals
real(dk)  ::  a,e,i,O,w,M,p
real(dk)  ::  EA
real(dk)  ::  sini,cosi,sinO,cosO,sinw,cosw,sinE,cosE,sinf,cosf
real(dk)  ::  R_pf(1:3),V_pf(1:3)
real(dk)  ::  ROT(1:3,1:3)

! ==============================================================================

! CLASSICAL ORBITAL ELEMENTS
! N.B: Units of length have to be consistent between COE(1) and GM; COE(2:6) are
! in radians.
!
! COE(1): a, semi-major axis
! COE(2): e, eccentricity
! COE(3): i, inclination
! COE(4): O, right ascension of ascending node
! COE(5): w, argument of periapsis
! COE(6): M, mean anomaly

! Modification (15.01.2016): tolKE is the tolerance for the iterative solver of
! the Kepler equation. If not present, it is set to the machine epsilon

! ==============================================================================================
! 01. UNPACKING OF STATE VECTOR AND CHECK FOR SPECIAL CASES
! ==============================================================================================

a = COE(1); e = COE(2); i = COE(3)
O = COE(4); w = COE(5); M = COE(6)
p = a*(1._dk - e**2)

! Bring M to [-2pi,2pi]
M = mod(M,real(twopi,dk))

! Elliptical, equatorial (assume that 'w' is the true longitude of periapsis)
!if (abs(sin(i)) <= zero) then
!! Circular, inclined (assume that 'nu' is the argument of latitude)
!if (e <= zero) w = 0._dk
!! The case "Circular, equatorial" is covered now, assuming that 'nu' is the true longitude.

sini = sin(i); cosi = cos(i)
sinO = sin(O); cosO = cos(O)
sinw = sin(w); cosw = cos(w)
! Compute true anomaly
if (present(tolKE)) then
    EA = KESOLVE(e,M,tolKE)
else
    EA = KESOLVE(e,M,mzero)
end if
sinE = sin(EA); cosE = cos(EA)
sinf = (sqrt(1._dk - e**2)*sinE)/(1._dk - e*cosE)
cosf = (cosE - e)/(1._dk - e*cosE)

! ==============================================================================
! 02. COMPUTE POS, VEL IN PERifOCAL FRAME
! ==============================================================================

R_pf = p*[cosf/(1._dk+e*cosf), sinf/(1._dk+e*cosf), 0._dk]
V_pf = sqrt(GM/p)*[-sinf, e + cosf, 0._dk]

! ==============================================================================
! 03. COMPUTE ROTATION MATRIX
! ==============================================================================

ROT(1,1) =  cosO*cosw - sinO*sinw*cosi
ROT(2,1) =  sinO*cosw + cosO*sinw*cosi
ROT(3,1) =  sinw*sini

ROT(1,2) = -cosO*sinw - sinO*cosw*cosi
ROT(2,2) = -sinO*sinw + cosO*cosw*cosi
ROT(3,2) =  cosw*sini

ROT(1,3) =  sinO*sini
ROT(2,3) = -cosO*sini
ROT(3,3) =  cosi

! ==============================================================================================
! 04. ROTATE VECTORS TO INERTIAL FRAME
! ==============================================================================================

R = matmul(ROT,R_pf)
V = matmul(ROT,V_pf)

end subroutine COE2CART


subroutine CART2COE(R,V,COE,mu)
! Description:
!    Computes classical orbital elements COE from Cartesian coordinates R_in,V_in.
!    Units have to be consistent.
!
! ==============================================================================

! VARIABLES
implicit none
! Arguments
real(dk),intent(in)   ::  R(1:3),V(1:3)
real(dk),intent(in)   ::  mu
real(dk),intent(out)  ::  COE(1:6)
! Locals
real(dk)              ::  p,a,e             ! Orbital elements
real(dk)              ::  inc,RAAN,AoP,nu,M ! Orbital elements
real(dk)              ::  En                ! Total energy
real(dk)              ::  h(1:3),n(1:3)     ! Angular momentum and nodal vector
real(dk)              ::  ecc(1:3)          ! Eccentricity vector
real(dk)              ::  sinn,cosn
real(dk)              ::  EA,sinE,cosE
real(dk)              ::  GA                ! Gudermannian anomaly
real(dk)              ::  rmag,nmag,hmag
real(dk)              ::  zero

! ==============================================================================
!
!                                            EXECUTION
!
! ==============================================================================

! CLASSICAL ORBITAL ELEMENTS
! COE(1): a,  semi-major axis
! COE(2): e,  eccentricity
! COE(3): i,  inclination
! COE(4): Om, right ascension of ascending node
! COE(5): w,  argument of periapsis OR longitude of periapsis
! COE(6): M,  mean anomaly

! =============================================================================
! 02. COMPUTE p,a,ecc,e
! =============================================================================
zero = 2._dk*mzero

rmag   = sqrt(dot_product(R,R))

h(:)   = CROSS_PRODUCT(R,V)
n(:)   = CROSS_PRODUCT([0._dk,0._dk,1._dk],h)
ecc(:) = ((DOT_PRODUCT(V,V) - mu/rmag)*R - DOT_PRODUCT(R,V)*V)/mu
e      = sqrt(dot_product(ecc,ecc))

nmag   = sqrt(dot_product(n,n))

En     = 0.5_dk*DOT_PRODUCT(V,V) - mu/rmag

if (ABS(En) > zero) then
    a  = -mu/(2._dk*En)
    p  = a*(1._dk-e**2)

else ! Exactly parabolic orbit, should be extremely rare
    a  = HUGE(0._dk)
    p  = DOT_PRODUCT(h,h)

end if

! =============================================================================
! 03. COMPUTE inc,RAAN,AoP,nu
! =============================================================================

hmag = sqrt(dot_product(h,h))
inc  = acos(h(3)/hmag)
RAAN = ACOS(n(1)/nmag); if (n(2) < 0._dk) RAAN = twopi - RAAN
AoP  = ACOS(DOT_PRODUCT(n,ecc)/(nmag*e)); if (ecc(3) < 0._dk) AoP  = twopi - AoP

nu   = ACOS(DOT_PRODUCT(ecc,R)/e/rmag)
if (DOT_PRODUCT(R,V) < 0._dk) then
    nu   = twopi - nu

end if

! SPECIAL CASES:
! Elliptical, equatorial. AoP = true longitude of periapsis
if (ABS(ABS(h(3)/hmag)-1._dk) < zero) then
    RAAN = 0._dk
    AoP  = ACOS(ecc(1)/e); if (ecc(2) < 0._dk) AoP = twopi - AoP
end if
! Circular, inclined.   nu = argument of latitude
if (e <= zero) then
    AoP = 0._dk
    nu  = ACOS(DOT_PRODUCT(n,R)/nmag/rmag); if (r(3) < 0._dk) nu = twopi - nu
    M = nu
end if
! Circular, equatorial. nu = true longitude.
if (ABS(ABS(h(3)/hmag)-1._dk) < zero .AND. e <= zero) then
    RAAN = 0._dk
    AoP  = 0._dk
    nu   = ACOS(R(1)/rmag); if (r(2) < 0._dk) nu = twopi - nu
end if

! Mean anomaly (see Battin, Ch. 4)
sinn = sin(nu); cosn = cos(nu)
if (e < 1._dk) then
  sinE = (sinn*sqrt(1._dk - e**2))/(1._dk + e*cosn)
  cosE = (e + cosn)/(1._dk + e*cosn)
  EA = atan2(sinE,cosE)
  M = EA - e*sinE

else if (e > 1._dk) then
  GA = 2._dk * atan( sqrt( (e - 1._dk) / (e + 1._dk) ) * tan(0.5_dk * nu) )
  M = e * tan(GA) - log( tan( 0.5_dk * GA + 0.25_dk * pi ))
  
end if

! Wrap M to [0,360]
M = mod(M + twopi,twopi)

COE = [a,e,inc,RAAN,AoP,M]

contains

function CROSS_PRODUCT(a,b)
! Description:
!    Computes cross product of a*b.
! ==============================================================================
!                                            PREAMBLE
! ==============================================================================

! VARIABLES
implicit none
! Function definition
real(dk)            :: CROSS_PRODUCT(1:3)
! Arguments
real(dk)            :: a(1:3),b(1:3)

! ==============================================================================
!                                            EXECUTION
! ==============================================================================
CROSS_PRODUCT(1) = a(2)*b(3) - a(3)*b(2)
CROSS_PRODUCT(2) = a(3)*b(1) - a(1)*b(3)
CROSS_PRODUCT(3) = a(1)*b(2) - a(2)*b(1)
end function CROSS_PRODUCT

end subroutine CART2COE


end module CART_COE
