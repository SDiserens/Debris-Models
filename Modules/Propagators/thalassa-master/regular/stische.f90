module STI_SCHE
! Description:
!    Contains procedures necessary for the Stiefel-Scheifele formulation. The
!    procedures evaluate the right-hand-side of the Stiefel-Scheifele equations
!    (STISCHE_RHS), provide an event function for LSODAR (STISCHE_EVT), and
!    provide coordinate and time transformations (STISCHE_TE2TIME, STISCHE2CART,
!    CART2STISCHE).
!
! References:
!    [1] Stiefel, E. L. and Scheifele, G. "Linear and Regular Celestial Mechanics",
!     Springer-Verlag, 1971.
!
! Author:
!    Davide Amato
!    Space Dynamics Group - Technical University of Madrid
!    The University of Arizona
!    davideamato@email.arizona.edu
!
! Revisions:
!     180608: change interface to PERT_EJ2K to add iF107.
!     180801: change call to perturbation routine, PERT_EJ2K. Eliminate needless
!             use associations. Refine comments.
!     190110: Add check for collision with the Moon.
!      
! ==============================================================================

use KINDS, only: dk
implicit none


contains


! ==============================================================================
! 01. INTEGRATION PROCEDURES
! ==============================================================================


subroutine STISCHE_RHS(neq,phi,z,zdot)
! Description:
!    Computes the value of the right-hand side of the equations of motion of the
!    EDromo formulation.
!
! Author:
!    Davide Amato
!    The University of Arizona
!    davideamato@email.arizona.edu
!
! Revisions:
!     180608: change interface to PERT_EJ2K to add iF107.
!     180801: change call to perturbation routine, PERT_EJ2K. Eliminate needless
!             use associations.
!
! ==============================================================================

! ! MODULES
use SETTINGS,      only: eqs,insgrav,isun,imoon,idrag,iF107,iSRP
use PHYS_CONST,    only: GE_nd
use PERTURBATIONS, only: PERT_EJ2K

! VARIABLES
implicit none

! Arguments
integer,intent(in)    ::  neq              ! Number of equations
real(dk),intent(in)   ::  phi              ! StiSche independent variable
real(dk),intent(in)   ::  z(1:neq)         ! StiSche state vector, ND
real(dk),intent(out)  ::  zdot(1:neq)      ! RHS of EoM's, ND

! Auxiliary quantities
real(dk)  ::  sph2,cph2
real(dk)  ::  aux(1:4),lte1,lte2,lte3
integer   ::  flag_time

! K-S state vector and its derivative
real(dk)  ::  u(1:4)
real(dk)  ::  du(1:4)

! State
real(dk)  ::  rV(1:3),vV(1:3),t
! real(dk)  ::  x_vec(1:3),y_vec(1:3)
real(dk)  ::  rmag,vsq

! Perturbations
real(dk)  ::  Vpot,mdVdr(1:3),dVdu(1:4),dVdt  ! Perturbing potential and its derivatives
real(dk)  ::  p(1:3),Lp(1:4)             ! Non-conservative perturbing accelerations

! ==============================================================================

! STATE VECTOR DICTIONARY

! z(1:4) = alpha
! z(5:8) = beta
! z(9)   = omega
! z(10)  = time (flag_time = 0) or time element (flag_time = 1)

! ==============================================================================
! 01. AUXILIARY QUANTITIES (1)
! ==============================================================================

! Store trig functions
sph2 = sin(0.5_dk*phi)
cph2 = cos(0.5_dk*phi)

! ==============================================================================
! 02. POSITION AND VELOCITY IN INERTIAL FRAME, TIME
! ==============================================================================

! K-S vector, Eq. (19,50)
u = z(1:4)*cph2 + z(5:8)*sph2
! Derivative of K-S vector, Eq. (19,51)
du = .5_dk*(-z(1:4)*sph2 + z(5:8)*cph2)

! Position in inertial frame. Eq. (19,56).
rV = [u(1)**2 - u(2)**2 - u(3)**2 + u(4)**2,&
      2._dk*(u(1)*u(2) - u(3)*u(4))        ,&
      2._dk*(u(1)*u(3) + u(2)*u(4))         ]
rmag = dot_product(u,u)

! Velocity in inertial frame. Eq. (19,58)
vV = 4._dk*z(9)/rmag*&
    [u(1)*du(1) - u(2)*du(2) - u(3)*du(3) + u(4)*du(4),  &
     u(2)*du(1) + u(1)*du(2) - u(4)*du(3) - u(3)*du(4),  &
     u(3)*du(1) + u(4)*du(2) + u(1)*du(3) + u(2)*du(4)]
vsq = dot_product(vV,vV)

flag_time = eqs - 7

! Get time
if ( flag_time == 0 ) then
    ! Physical time
    t = z(10)
elseif  ( flag_time == 1 ) then
    ! Linear time element
    t = z(10) - dot_product(u,du)/z(9)
end if

! ==============================================================================
! 03. PERTURBING POTENTIAL
! ==============================================================================

! Initialize
Vpot = 0._dk; dVdt = 0._dk; mdVdr = 0._dk
! Evaluate potential perturbations
call PERT_EJ2K(insgrav,0,0,0,0,0,rV,vV,rmag,t,mdVdr,Vpot,dVdt)

! ==============================================================================
! 04. PERTURBING ACCELERATIONS
! ==============================================================================

! Initializations
p = 0._dk
call PERT_EJ2K(0,isun,imoon,idrag,iF107,iSRP,rV,vV,rmag,t,p)

! ==============================================================================
! 05. COMPUTE AUXILIARY QUANTITIES (2)
! ==============================================================================

! Eq. (19,60)
Lp = [ u(1)*p(1) + u(2)*p(2) + u(3)*p(3),  &
      -u(2)*p(1) + u(1)*p(2) + u(4)*p(3),  &
      -u(3)*p(1) - u(4)*p(2) + u(1)*p(3),  &
       u(4)*p(1) - u(3)*p(2) + u(2)*p(3)   ]

! Eq. (9,44) - note that mdVdr = -dV/dr
dVdu = -2._dk*[u(1)*mdVdr(1) + u(2)*mdVdr(2) + u(3)*mdVdr(3),  &
              -u(2)*mdVdr(1) + u(1)*mdVdr(2) + u(4)*mdVdr(3),  &
              -u(3)*mdVdr(1) - u(4)*mdVdr(2) + u(1)*mdVdr(3),  &
               u(4)*mdVdr(1) - u(3)*mdVdr(2) + u(2)*mdVdr(3)]

! ==============================================================================
! 06. COMPUTE RIGHT-HAND SIDE
! ==============================================================================

! Eq. (19,61)
zdot(9) = -rmag/(8._dk*z(9)**2)*dVdt - .5_dk/z(9)*dot_product(du,Lp)

! Eq. (19,63)
aux = (.5_dk/z(9)**2) * (.5_dk*Vpot*u + rmag/4._dk * (dVdu - 2._dk*Lp)) + &
    &   2._dk/z(9) * zdot(9) * du 

zdot(1:4) =  aux*sph2
zdot(5:8) = -aux*cph2

! Time / Time Element
if (flag_time == 0) then
   ! Generalized Sundman transformation
   zdot(10) = .5_dk*rmag/z(9)

else if (flag_time == 1) then   ! Linear Time Element
   ! Eq. (19,62)
   lte1 = (GE_nd - 2._dk*rmag*Vpot)/(8._dk*z(9)**3)
   lte2 = (rmag/(16._dk*z(9)**3)) * dot_product(u,dVdu - 2._dk*Lp)
   lte3 = (2._dk/z(9)**2) * zdot(9) * dot_product(u,du)
   zdot(10) = lte1 - lte2 - lte3

end if

end subroutine STISCHE_RHS


subroutine STISCHE_EVT(neq,phi,z,ng,roots)
! Description:
!    Finds roots to stop the integration for the SS formulation.
! 
! Revisions:
!    190110: Add header comments. Add check for collision with the Moon.
! 
! ==============================================================================

! MODULES
use SUN_MOON,    only: EPHEM
use AUXILIARIES, only: MJD0,MJDnext,MJDf,DU,TU
use PHYS_CONST,  only: secsPerDay,RE,reentry_radius_nd,ReqM
use SETTINGS,    only: eqs, imoon, imcoll

! VARIABLES
implicit none
! Arguments IN
integer,intent(in)    ::  neq
integer,intent(in)    ::  ng
real(dk),intent(in)   ::  phi
real(dk),intent(in)   ::  z(1:neq)
! Arguments OUT
real(dk),intent(out)  ::  roots(1:ng)

! Locals
integer   ::  flag_time
real(dk)  ::  t                   ! Current time [-]
real(dk)  ::  sph2,cph2,rmag, dmag
real(dk)  ::  r_vec(1:3), v_vec(1:3), rMoon(1:3), vMoon(1:3)
real(dk)  ::  u(1:4) 

! ==============================================================================

roots = 1._dk

! Get time
flag_time = eqs - 7
t = STISCHE_TE2TIME(z,phi,flag_time)

! ==============================================================================
! 01. Next timestep
! ==============================================================================

roots(1) = t - (MJDnext - MJD0)*secsPerDay*TU

! ==============================================================================
! 02. Stop integration
! ==============================================================================

roots(2) = t - (MJDf - MJD0)*secsPerDay*TU

! ==============================================================================
! 03. Earth re-entry
! ==============================================================================
sph2 = sin(phi/2._dk)
cph2 = cos(phi/2._dk)
u    = z(1:4)*cph2 + z(5:8)*sph2

rmag = dot_product(u,u)

roots(3) = rmag - reentry_radius_nd

! ==============================================================================
! 04. Moon collision (only active when Moon is present)
! ==============================================================================

! All quantities are dimensional. Accuracy could be improved by using
! dimensionless quantities, but this would require
! non-dimensionalizing ReqM in some part of the code, which would worsen the
! code reliability.
roots(4) = 1.
if (imoon > 0 .and. (imcoll /= 0)) then
  call STISCHE2CART(phi,z,r_vec,v_vec)
  r_vec = r_vec * DU
  call EPHEM(2, 1._dk, 1._dk, t, rMoon, vMoon)
  dmag = sqrt(dot_product( (r_vec - rMoon), (r_vec - rMoon) ) )
  roots(4) = dmag - ReqM

end if

end subroutine STISCHE_EVT

! ==============================================================================
! 02. TRANSFORMATIONS AND PROCESSING PROCEDURES
! ==============================================================================


function STISCHE_TE2TIME(z,phi,flag_time)
! Description:
!    Gets the value of physical time from the Stiefel-Scheifele state vector "z" and
!    fictitious time "phi".
!
! ==============================================================================

! VARIABLES
implicit none
! Arguments IN
real(dk),intent(in)  ::  z(1:10),phi
integer,intent(in)   ::  flag_time
! Locals
real(dk)             ::  alphaSq,betaSq,alphaDotBeta
! Function definition
real(dk)  ::  STISCHE_TE2TIME

! ==============================================================================

if ( flag_time == 0 ) then
    ! Physical time
    STISCHE_TE2TIME = z(10)

else if  ( flag_time == 1 ) then
    ! Linear time element. This is derived by plugging Eq. (19, 55) in
    ! Eq. (19,59)
    alphaSq = dot_product(z(1:4),z(1:4))
    betaSq  = dot_product(z(5:8),z(5:8))
    alphaDotBeta = dot_product(z(1:4),z(5:8))
    STISCHE_TE2TIME = z(10) +&
    &.5_dk*( (alphaSq-betaSq)/2._dk * sin(phi) - alphaDotBeta * cos(phi) )/z(9)
    
end if

end function STISCHE_TE2TIME


subroutine CART2STISCHE(R,V,t0,mu,DU,TU,z,phi,W,flag_time)
! Description:
!    Converts from Cartesian coordinates to Stiefel-Scheifele elements. It requires
!    the current values of the fictitious time, perturbing potential and initial
!    time.
!
!===============================================================================

! VARIABLES
implicit none

! Arguments
integer,parameter     ::  neq = 10	  ! Number of elements of Stiefel-Scheifele state vector
real(dk),intent(in)   ::  R(1:3),V(1:3)	  ! Dimensional position and velocity [km,km/s]
real(dk),intent(in)   ::  mu              ! Gravitational parameter (dimensional)
real(dk),intent(in)   ::  DU,TU           ! Ref. quantities for non-dimensionalization
real(dk),intent(in)   ::  phi             ! Initial phi value
real(dk),intent(in)   ::  t0              ! Initial time value [s]
real(dk),intent(in)   ::  W               ! Potential energy [km^2/s^2]
integer,intent(in)    ::  flag_time	  ! = 0 physical time
         				  ! = 1 constant time element
					  ! = 2 linear time element
real(dk),intent(out)  ::  z(1:neq)        ! Stiefel-Scheifele state vector

! Local variables
real(dk)  ::  x(1:3)           		        ! Cartesian position, ND
real(dk)  ::  xdot(1:3)                     ! Cartesian velocity, ND
real(dk)  ::  Ksq                           ! Grav parameter, ND
real(dk)  ::  Rmag,Vmag 	                ! Radius and velocity magnitudes, ND
real(dk)  ::  pot0           		        ! Potential, ND
real(dk)  ::  totEn0           		        ! Initial total energy, ND
real(dk)  ::  cph2,sph2
real(dk)  ::  u_vec(1:4),du_vec(1:4)            ! K-S parameters and their derivatives
real(dk)  ::  GE                	        ! Auxiliary variable
real(dk)  ::  zero             		        ! Reference machine zero

! ==============================================================================

! ==============================================================================
! 01. NON-DIMENSIONALIZATION
! ==============================================================================
x(1:3)    = R/DU
xdot(1:3) = V/(DU*TU)
pot0      = W/(DU*TU)**2
Ksq       = mu/(DU**3*TU**2)

! ==============================================================================
! 02. AUXILIARY QUANTITIES
! ==============================================================================
! Compute machine zero for comparisons
zero = epsilon(0._dk)

Rmag = sqrt(dot_product(x(1:3),x(1:3)))
cph2 = cos(phi/2._dk)
sph2 = sin(phi/2._dk)

! ==============================================================================
! 03. COMPUTE z(1) - z(9)
! ==============================================================================
! Total energy
totEn0  = .5_dk*dot_product(xdot,xdot) - Ksq/Rmag + pot0
z(9) = sqrt(-totEn0/2._dk)

! K-S parameters
if ( x(1) >= 0._dk ) then
   u_vec(1) = 0._dk
   u_vec(4) = sqrt(.5_dk*(Rmag + x(1)) - u_vec(1)**2)
   u_vec(2) = (x(2)*u_vec(1) + x(3)*u_vec(4))/(Rmag + x(1))
   u_vec(3) = (x(3)*u_vec(1) - x(2)*u_vec(4))/(Rmag + x(1))
else
   u_vec(2) = 0._dk
   u_vec(3) = sqrt(.5_dk*(Rmag - x(1)) - u_vec(2)**2)
   u_vec(1) = (x(2)*u_vec(2) + x(3)*u_vec(3))/(Rmag - x(1))
   u_vec(4) = (x(3)*u_vec(2) - x(2)*u_vec(3))/(Rmag - x(1))
end if
! Derivatives of the K-S parameters wrt the independent variable
du_vec(1) = ( u_vec(1)*xdot(1) + u_vec(2)*xdot(2) + u_vec(3)*xdot(3))/(4._dk*z(9))
du_vec(2) = (-u_vec(2)*xdot(1) + u_vec(1)*xdot(2) + u_vec(4)*xdot(3))/(4._dk*z(9))
du_vec(3) = (-u_vec(3)*xdot(1) - u_vec(4)*xdot(2) + u_vec(1)*xdot(3))/(4._dk*z(9))
du_vec(4) = ( u_vec(4)*xdot(1) - u_vec(3)*xdot(2) + u_vec(2)*xdot(3))/(4._dk*z(9))

z(1:4) = cph2*u_vec - 2._dk*sph2*du_vec
z(5:8) = sph2*u_vec + 2._dk*cph2*du_vec

! ==============================================================================
! 04. TIME / TIME ELEMENT
! ==============================================================================
if ( flag_time == 0 ) then
    ! Physical time
    z(10) = t0*TU

elseif  ( flag_time == 1 ) then
    ! Linear time element
    z(10) = t0*TU + dot_product(u_vec,du_vec)/z(9)

end if

end subroutine CART2STISCHE


subroutine STISCHE2CART(phi,z,r,v)
! Description:
!    Transforms from the Stiefel-Scheifele state vector "z" and fictitious time
!    "phi" to Cartesian position and velocity "r", "v".
!    **All quantities are dimensionless, unlike in CART2STISCHE**.
!
! ==============================================================================

! VARIABLES
implicit none
! Arguments IN
real(dk),intent(in)   ::  z(1:10),phi
! Arguments OUT
real(dk),intent(out)  ::  r(1:3),v(1:3)

! Auxiliaries
real(dk)  ::  sph2,cph2
real(dk)  ::  rmag
real(dk)  ::  u(1:4),du(1:4)

! ==============================================================================

! ==============================================================================
! 01. AUXILIARY QUANTITIES
! ==============================================================================

! Store trig functions
sph2 = sin(phi/2._dk)
cph2 = cos(phi/2._dk)

! ==============================================================================
! 02. POSITION IN INERTIAL FRAME
! ==============================================================================

! K-S state vector and its derivative
u  = z(1:4)*cph2 + z(5:8)*sph2
du = .5_dk*( -z(1:4)*sph2 + z(5:8)*cph2)

! Position in inertial frame
r = [u(1)**2 - u(2)**2 - u(3)**2 + u(4)**2,  &
     2._dk*(u(1)*u(2) - u(3)*u(4)),  &
     2._dk*(u(1)*u(3) + u(2)*u(4))]
rmag = u(1)**2 + u(2)**2 + u(3)**2 + u(4)**2

! ==============================================================================
! 03. VELOCITY IN INERTIAL FRAME
! ==============================================================================

v = 4._dk*z(9)/rmag*&
    [u(1)*du(1) - u(2)*du(2) - u(3)*du(3) + u(4)*du(4),  &
     u(2)*du(1) + u(1)*du(2) - u(4)*du(3) - u(3)*du(4),  &
     u(3)*du(1) + u(4)*du(2) + u(1)*du(3) + u(2)*du(4)]

end subroutine STISCHE2CART


end module STI_SCHE
