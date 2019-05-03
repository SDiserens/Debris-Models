module SRP
! Description:
!    Contains subroutines to compute the perturbing acceleration due to SRP.
!
! Author:
!    Davide Amato
!    Space Dynamics Group - Technical University of Madrid
!    The University of Arizona
!    davideamato@email.arizona.edu
!
! Revisions:
!     180609: add calculation of eclipse conditions.
! 
! ==============================================================================

! MODULES
use KINDS, only: dk
implicit none




contains




function SRP_ACC(iSRP,sunRadius,occRadius,pSRP_1au,au,CR,A2M_SRP,r,r_sun)
! Description:
!    Computes the perturbing acceleration due to solar radiation pressure.
!    Input units have to be consistent.
!
! Author:
!    Davide Amato
!    The University of Arizona
!    davideamato@email.arizona.edu
!
! Revisions:
!     180609: add call to ECLIPSE, change arguments.
! 
! ==============================================================================

! VARIABLES
implicit none
! Arguments IN
integer,intent(in)   ::  iSRP
real(dk),intent(in)  ::  occRadius,sunRadius
real(dk),intent(in)  ::  pSRP_1au,au,CR,A2M_SRP,r(1:3),r_sun(1:3)
! Function definition
real(dk)  ::  SRP_ACC(1:3)

! Locals
real(dk)  :: nu    ! Shadowing factor
real(dk)  :: pSRP
real(dk)  :: dSun(1:3),dSunNorm

! ==============================================================================

! Sun->Spacecraft vector
dSun = r - r_sun
dSunNorm = sqrt(dot_product(dSun,dSun))

! SRP constant at the current distance from the Sun
pSRP = pSRP_1au*(au/dSunNorm)**2

! Perturbing acceleration
nu = 1._dk
if (iSRP == 2) then
  nu = ECLIPSE(sunRadius,occRadius,r,r_sun)

end if
SRP_ACC = nu * pSRP * CR * A2M_SRP * (dSun/dSunNorm)

end function SRP_ACC




function ECLIPSE(sunRadius,occRadius,r,rSun)
! Description:
!    Find out if the Sun is eclipsed by an occulting body. Returns value of
!    the coefficient \nu in [1], that is:
!    \nu = 1                   ,  for orbiter in sunlight
!    \nu = 0                   ,  for orbiter in umbra (total eclipse)
!    \nu = 1 - A/(\pi * a^2)   , for orbiter in penumbra (partial eclipse)
!    \nu = 1 - b^2/a^2         , for orbiter in antumbra (annular eclipse)
!    
! Author:
!    Davide Amato
!    The University of Arizona
!    davideamato@email.arizona.edu
!
! Revisions:
!     180609: procedure created.
!
! References:
!     [1] Montenbruck, O., and Gill, E. 'Satellite Orbits', Springer, 2000.
! 
! ==============================================================================

! Modules
use PHYS_CONST, only: pi

! Variables
implicit none
real(dk),intent(in)  ::  r(1:3)       ! Occulting body -> orbiter
real(dk),intent(in)  ::  rSun(1:3)    ! Occulting body -> Sun
real(dk),intent(in)  ::  sunRadius,occRadius  ! Physical radii of the bodies 
! Returns
real(dk)  ::  ECLIPSE

! Locals
real(dk)  ::  rOrbSun(1:3)
real(dk)  ::  rDotRSun,rNorm,rSunNorm,rOrbSunNorm
real(dk)  ::  aApp,bApp  ! Apparent radii of the Sun and occulting body
real(dk)  ::  cApp       ! Apparent separation of the centers
real(dk)  ::  x,y        ! Aux quantities for the calculation of the occulted area
real(dk)  ::  occArea
real(dk)  ::  nu         ! Shadowing factor

! ==============================================================================

nu = 1._dk

rDotRSun = dot_product(r,rSun)
! Check if orbiter is in opposition
if (rDotRSun < 0._dk) then
  rNorm    = sqrt(dot_product(r,r))
  rSunNorm = sqrt(dot_product(rSun,rSun))

  ! Apparent radii of the bodies
  aApp = asin(sunRadius / rSunNorm)
  bApp = asin(occRadius / rNorm)
  
  rOrbSun     = rSun - r
  rOrbSunNorm = sqrt(dot_product(rOrbSun,rOrbSun))
  
  ! Apparent separation of the centers
  cApp = acos( - dot_product( r, rOrbSun ) / ( rNorm * rOrbSunNorm ) )
  
  ! Shadow condition
  if (aApp + bApp >= cApp) then
    if (cApp > abs(bApp - aApp)) then
      ! Penumbra
      x = ( cApp**2 + aApp**2 - bApp**2 ) / (2 * cApp)
      y = sqrt( aApp**2 - x**2 )

      occArea = aApp ** 2 * acos( x / aApp ) + &
      & bApp ** 2 * acos( (cApp - x)/bApp ) - cApp * y

      nu = 1._dk - occArea / (pi * aApp ** 2)
    
    else if ( (cApp <= abs(bApp - aApp) .and. bApp > aApp ) ) then
      ! Umbra
      nu = 0._dk
    
    else if ( (cApp <= abs(bApp - aApp) .and. bApp <= aApp ) ) then
      ! Antumbra
      nu = 1._dk - bApp**2 / aApp**2

    end if

  end if

end if

ECLIPSE = nu

end function ECLIPSE




end module SRP