module NSGRAV
! Description:
!    Contains procedures for the calculation of perturbations due to the non-
!    sphericity of the main body. INITIALIZE_NSGRAV reads main body data from a
!    data file, and initializes coefficient matrices.
!    The calculation of the perturbing potential, perturbing acceleration, and
!    the time derivative of the potential in the body-fixed frame (the latter is
!    needed in regularized formulations) takes place in PINES_NSG.
!    NORMFACT gives the normalization factor for the gravitational coefficients.
! 
! Author:
!    Davide Amato
!    The University of Arizona
!    davideamato@email.arizona.edu
! 
! Revisions:
!    180806: Overhaul and implementation of Pines method.
!    181204: Use GMST (IERS 2006 conventions) rather than ERA. Consider Earth
!            Rotation Rate derived from IERS 2006 conventions.
! 
! ==============================================================================

! MODULES
use KINDS,      only: dk
use SETTINGS,   only: gdeg,gord
use IO,         only: id_earth
use PHYS_CONST, only: qk,GE,RE,flatt,omegaE,secsPerDay,secsPerSidDay,twopi,&
&ERR_constant
implicit none

! VARIABLES
! Spherical harmonics (unnormalized) 
integer               ::  maxDeg, maxOrd
real(dk),allocatable  ::  Cnm(:,:),Snm(:,:)

! Pines algorithm arrays
real(dk),allocatable  ::  Anm(:,:),Dnm(:,:),Enm(:,:),Fnm(:,:),Gnm(:,:)
real(dk),allocatable  ::  Rm(:),Im(:),Pn(:)
real(dk),allocatable  ::  Aux1(:),Aux2(:),Aux3(:),Aux4(:,:)




contains




subroutine INITIALIZE_NSGRAV(earthFile)
! Description:
!    Reads Earth gravity data from a text file. Initializes the gravitational
!    parameter, equatorial radius, flattening, rotational velocity, spherical
!    harmonics coefficients, and auxiliary matrices for Pines' algorithm. The
!    latter computes the Earth potential in Cartesian coordinates, avoiding
!    singularities due to the spherical harmonics.
!    Part of this subroutine is due to Hodei Urrutxua (Universidad Rey Juan
!    Carlos, Madrid, Spain) and Claudio Bombardelli (Universidad Politécnica de
!    Madrid, Madrid, Spain).
! 
! Author:
!    Davide Amato
!    The University of Arizona
!    davideamato@email.arizona.edu
! 
! Revisions:
!    180806: Subroutine created from part of READ_PHYS().
! 
! ==============================================================================

! Arguments
character(len=*),intent(in)  ::  earthFile

! Locals
character(len=72)  ::  dummy
real(dk)           ::  invFlatt
integer            ::  i,j,l,m,n


! ==============================================================================

open(unit=id_earth,file=trim(earthFile),status='old',action='read')
read(id_earth,'(a)') (dummy, i=1,4)
read(id_earth,'(a37,i3)') dummy, maxDeg
read(id_earth,'(a37,i3)') dummy, maxOrd
read(id_earth,'(a36,e22.15)') dummy, GE
read(id_earth,'(a36,e22.15)') dummy, RE
read(id_earth,'(a36,e22.15)') dummy, invFlatt
read(id_earth,'(a36,e22.15)') dummy, omegaE

flatt    = 1._dk/invFlatt

! Read and de-normalize spherical harmonics coefficients and auxiliary matrices
! for Pines' algorithm.
! Initialize
l = 1; m = 0;
allocate(Cnm(1:maxDeg,0:maxDeg)); Cnm = 0._dk
allocate(Snm(1:maxDeg,0:maxDeg)); Snm = 0._dk
allocate(Anm(0:maxDeg+2, 0:maxDeg+2))
allocate(Dnm(1:maxDeg,   0:maxDeg))
allocate(Gnm(1:maxDeg,   0:maxDeg))
allocate(Enm(1:maxDeg,   1:maxDeg))
allocate(Fnm(1:maxDeg,   1:maxDeg))
allocate(Rm(0:maxDeg)    )
allocate(Im(0:maxDeg)    )
allocate(Pn(0:maxDeg + 1))
allocate(Aux1(1:maxDeg+1))
allocate(Aux2(1:maxDeg+1))
allocate(Aux3(1:maxDeg+1))
allocate(Aux4(1:maxDeg+1, 0:maxDeg+1))

read(id_earth,'(a)') (dummy, i=1,2)
do i=1,maxDeg
  do j=0,minval([i,maxOrd])
    read(id_earth,'(2(1x,i2),2(1x,e24.17))') l,m,Cnm(i,j),Snm(i,j)
    Cnm(i,j) = Cnm(i,j)/NORMFACT(i,j)
    Snm(i,j) = Snm(i,j)/NORMFACT(i,j)
  end do
end do

close(id_earth)

! Fill coefficient arrays for Pines algorithm
Anm(:,:) = 0._dk
Dnm(:,:) = 0._dk
Gnm(:,:) = 0._dk
Enm(:,:) = 0._dk
Fnm(:,:) = 0._dk
Anm(0,0) = 1._dk
Anm(1,1) = 1._dk
do n = 1,maxDeg + 1
  Aux1(n) = 2._dk*n + 1._dk
  Aux2(n) = Aux1(n) / (n+1._dk)
  Aux3(n) = n / (n+1._dk)
  do m = 0, n-1
    Aux4(n,m) = (n+m+1._dk)
  end do
end do

! Earth Rotation Rate (revolutions per tropical day)
secsPerSidDay = twopi/omegaE
ERR_constant = secsPerDay/secsPerSidDay

end subroutine INITIALIZE_NSGRAV




function NORMFACT(l,m)
! Description:
!    Normalization factor for the spherical harmonics coefficients and
!    associated Legendre functions:
! 
!    sqrt( (l + m)! / ( (2 - delta_{0,m}) * (2n  + 1) * (n - m)! ) )
! 
! Reference:
!    [1] Montenbruck, O., Gill, E., "Satellite Orbits", p. 58, Springer, 2000.
! 
! Author:
!    Davide Amato
!    The University of Arizona
!    davideamato@email.arizona.edu
! 
! ==============================================================================

! Arguments and function definition
integer,intent(in)  ::  l,m
real(qk)            ::  NORMFACT
! Locals
real(qk)            ::  lr,mr
real(qk)            ::  kron
real(qk)            ::  numer,denom

! ==============================================================================
lr = real(l,qk)
mr = real(m,qk)

numer = gamma(lr + mr + 1._qk)

if (m == 0) then
	kron = 1._qk
else
	kron = 0._qk
end if

denom = (2._qk - kron) * (2._qk*lr + 1._qk) * gamma(lr - mr + 1._qk)

NORMFACT = sqrt(numer/denom)

end function NORMFACT



subroutine PINES_NSG(GM,RE,rIn,tau,FIn,pot,dPot)
! Description:
!    Compute the perturbing acceleration, perturbing potential (optional), and
!    time derivative of the perturbing potential in the body-fixed frame
!    (optional), given the position vector wrt the non-spherical body, its
!    gravitational parameter and its potential coefficients.
!    Uses the method described in the Reference to perform the calculation in
!    Cartesian coordinates, thereby avoiding associated Legendre functions and
!    polynomials. The formulation is non-singular everywhere for r > RE, where
!    RE is the radius of the non-spherical body.
! 
!    Adapted from code developed by Hodei Urrutxua (Universidad Rey Juan Carlos,
!    Madrid, Spain) and Claudio Bombardelli (Universidad Politécnica de Madrid,
!    Madrid, Spain).
! 
! Author:
!    Davide Amato
!    The University of Arizona
!    davideamato@email.arizona.edu
! 
! Reference:
!    S. Pines, "Uniform Representation of the Gravitational Potential and its
!    derivatives," AIAA J. 11 (11), pp. 1508-1511, 1973.
! 
! Revisions:
!    180806: First working version of subroutine.
!    181204: Use GMST (IERS 2006 conventions) rather than ERA. Consider Earth
!            Rotation Rate derived from IERS 2006 conventions.
!
! ==============================================================================

! Use associations
use AUXILIARIES, only: MJD0, TU, T2MJD
use PHYS_CONST,  only: delta_JD_MJD
use PHYS_CONST,  only: ERR_IAU06, UTC2TT

! Arguments
real(dk),intent(in)   ::  GM             ! Gravitational parameter
real(dk),intent(in)   ::  RE             ! Equatorial radius
real(dk),intent(in)   ::  rIn(1:3)       ! Position in the inertial frame
real(dk),intent(in)   ::  tau            ! Physical time
real(dk),intent(out),optional  ::  FIn(1:3)  ! Perturbing acceleration in the inertial frame
real(dk),intent(out),optional  ::  pot   ! Perturbing potential
real(dk),intent(out),optional  ::  dPot  ! Time derivative of the potential in body-fixed frame
! Locals
real(dk)  ::  rNorm,rNormSq  ! Norm of position vector and square
real(dk)  ::  s,t,u  ! Direction cosines in the body-fixed frame
real(dk)  ::  rho    ! = equatorial radius / r 
real(dk)  ::  F(1:3),a1,a2,a3,a4  ! Acceleration and its components 
integer   ::  n,m    ! Harmonic indices
logical   ::  skip_EFG
! GMST-related quantities
real(dk)  ::  MJD_UTC, MJD_TT      ! UTC and TT dates
real(dk)  ::  GMST,cosGMST,sinGMST ! GMST and its trig functions
real(dk)  ::  ERR, ERR_nd          ! Earth Rotation Rate [rad/s, -]

! SOFA routines
real(dk) :: iau_GMST06

! ==============================================================================

rNormSq = dot_product(rIn,rIn)
rNorm = sqrt(rNormSq)

! ==============================================================================
! 01. Transform from inertial to body-fixed frame
! ==============================================================================

u = rIn(3)/rNorm

! Greenwich Mean Sidereal Time (IAU 2006 conventions)
MJD_UTC = T2MJD(tau)
MJD_TT  = UTC2TT(MJD_UTC)

GMST = iau_GMST06 ( delta_JD_MJD, MJD_UTC, delta_JD_MJD, MJD_TT )
cosGMST = cos(GMST); sinGMST = sin(GMST)

! Rotate equatorial components of rIn to get direction cosines in the body-fixed frame
s = (rIn(1) * cosGMST + rIn(2) * sinGMST)/rNorm
t = (rIn(2) * cosGMST - rIn(1) * sinGMST)/rNorm

rho = RE/rNorm

! ==============================================================================
! 02. Fill in coefficient matrices and auxiliary vectors
! ==============================================================================

! Fill A Matrix
Anm(0,0) = 1._dk
Anm(1,1) = 1._dk
Anm(1,0) = u
do n = 1, gdeg + 1
  Anm(n+1,n+1) = Aux1(n) * Anm(n,n) ! Fill the diagonal
  Anm(n+1,0) = Aux2(n) * u * Anm(n,0) - Aux3(n) * Anm(n-1,0) ! Fill the 1st column
  Anm(n+1,n) = u * Anm(n+1,n+1)    ! Fill the subdiagonal

end do
do n = 2, gdeg + 1 ! Fill remaining elements
  do m = 0, n - 2
    Anm(n+1,m+1) = Aux4(n,m) * Anm(n,m) + u * Anm(n,m+1)
  end do

end do

! Fill R, I, and P vectors
Rm(0) = 1._dk
Im(0) = 0._dk
Pn(0) = GM / rNorm
Pn(1) = rho * Pn(0)
do n = 1, gdeg
  Rm(n)   = s  * Rm(n-1) - t * Im(n-1)
  Im(n)   = s  * Im(n-1) + t * Rm(n-1)
  Pn(n+1) = rho * Pn(n)

end do

! Fill D, E, and F matrices
skip_EFG = present(pot) .and. .not.(present(FIn)) .and. .not.(present(dPot))
do m = 1, gord
  do n = m, gdeg
    Dnm(n,m) = Cnm(n,m)*Rm(m)   + Snm(n,m)*Im(m)
    if (.not.(skip_EFG)) then
      Enm(n,m) = Cnm(n,m)*Rm(m-1) + Snm(n,m)*Im(m-1)
      Fnm(n,m) = Snm(n,m)*Rm(m-1) - Cnm(n,m)*Im(m-1)

    end if
  end do

end do
do n = 1, gdeg
  Dnm(n,0) = Cnm(n,0)*Rm(0)  !+ S(n,0)*I(0) = 0
end do

! ==============================================================================
! 03. Perturbing potential
! ==============================================================================

if (present(pot)) then
  pot = 0._dk
  do m = 1, gord
    do n = m, gdeg
      pot = pot + Pn(n) * Anm(n,m) * Dnm(n,m)
		
    end do
	
  end do
  do n = 1, gdeg
  pot = pot + Pn(n) * Anm(n,0) * Dnm(n,0)
	
  end do
  pot = -pot  ! Change the sign to get the potential
end if

! ==============================================================================
! 04. Perturbing acceleration
! ==============================================================================

if (present(FIn)) then
  a1 = 0._dk; a2 = 0._dk; a3 = 0._dk; a4 = 0._dk
  do m = 1, gord
    do n = m, gdeg
      a1 = a1 + Pn(n+1) * Anm(n,m) * m * Enm(n,m)
      a2 = a2 + Pn(n+1) * Anm(n,m) * m * Fnm(n,m)
      a3 = a3 + Pn(n+1) * Anm(n,m+1)   * Dnm(n,m)
      a4 = a4 - Pn(n+1) * Anm(n+1,m+1) * Dnm(n,m)
    end do
  end do
  do n = 1, gdeg
    a3 = a3 + Pn(n+1) * Anm(n,1)     * Dnm(n,0)
    a4 = a4 - Pn(n+1) * Anm(n+1,1)   * Dnm(n,0)

  end do
  F = [a1, a2, a3] + [s, t, u] * a4
  F = F / RE

  ! Transform to inertial frame
  FIn(1) = F(1)*cosGMST - F(2)*sinGMST
  FIn(2) = F(1)*sinGMST + F(2)*cosGMST
  FIn(3) = F(3)

end if

! ==============================================================================
! 05. Time derivative of potential in body-fixed frame
! ==============================================================================

if(present(dPot)) then
  dPot = 0._dk
  ERR = ERR_IAU06(0._dk, MJD_TT)
  ERR_nd = ERR / TU
  do m = 1, gord
    do n = m, gdeg
      Gnm(n,m) = m * ( t * Enm(n,m) - s * Fnm(n,m) )
      Gnm(n,m) = ERR_nd * Gnm(n,m)
      dPot = dPot + Pn(n) * Anm(n,m) * Gnm(n,m)
      
    end do
    
  end do
  dPot = -dPot
end if

end subroutine PINES_NSG




end module NSGRAV
