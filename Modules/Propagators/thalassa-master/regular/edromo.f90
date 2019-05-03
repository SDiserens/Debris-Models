module EDROMO
! Description:
!    Contains procedures necessary for the EDromo formulation. The procedures
!    evaluate the right-hand side of the equations (EDROMO_RHS), provide an
!    event function for LSODAR (EDROMO_EVT), provide coordinate and time
!    transformations (EDROMO_TE2TIME, CART2EDROMO, EDROMO2CART,
!    INERT2ORB_EDROMO), and compute auxiliary quantities (EDROMO_PHI0).
!
! References:
! [1] Baù, G., Bombardelli, C., Peláez, J., and Lorenzini, E., "Nonsingular
!     orbital elements for special perturbations in the two-body problem".
!     MNRAS 454(3), pp. 2890-2908. 2015.
!
! Author:
!    Davide Amato
!    Space Dynamics Group - Technical University of Madrid
!    The University of Arizona
!    davideamato@email.arizona.edu
! 
! Revisions:
!    180801: change call to perturbation routine, PERT_EJ2K. Eliminate use
!            association with PHYS_CONST in EDROMO_RHS. Refine comments to the
!            module.
!    190110: Add check for collision with the Moon. Add optional flag to
!            EDROMO2CART to return position only.
!
! ==============================================================================

use KINDS, only: dk
implicit none


contains


! ==============================================================================
! 01. INTEGRATION PROCEDURES
! ==============================================================================


subroutine EDROMO_RHS(neq,phi,z,zdot)
! Description:
!    Computes the value of the right-hand side of the equations of motion of the
!    EDromo formulation.
!
! Author:
!    Davide Amato
!    Space Dynamics Group - Technical University of Madrid
!    The University of Arizona
!    davideamato@email.arizona.edu
!
! Revisions:
!     180608: change interface to PACC_EJ2K to add iF107.
!     180801: change call to perturbation routine, PERT_EJ2K. Eliminate use
!             association with PHYS_CONST.
!
! ==============================================================================

! MODULES
use SETTINGS,      only: eqs,insgrav,isun,imoon,idrag,iSRP,iF107
use PERTURBATIONS, only: PERT_EJ2K

! VARIABLES
implicit none

! Arguments
integer,intent(in)       ::  neq              ! Number of equations
real(dk),intent(in)      ::  phi              ! EDromo independent variable
real(dk),intent(in)      ::  z(1:neq)         ! EDromo state vector, ND
real(dk),intent(out)     ::  zdot(1:neq)      ! RHS of EoM's, ND

! Auxiliary quantities
real(dk)    ::  sph,cph
real(dk)    ::  rho,rmag,zeta,emme
real(dk)    ::  cnu,snu
real(dk)    ::  enne,L3,wz
real(dk)    ::  aux0,aux1,aux2,aux3,aux4
integer     ::  flag_time

! State
real(dk)    ::  rV(1:3),vV(1:3),t
real(dk)    ::  x_vec(1:3),y_vec(1:3)
real(dk)    ::  i_vec(1:3),j_vec(1:3)
real(dk)    ::  v_rad,v_tan,vmag,vsq
real(dk)    ::  cosg,sing

! Perturbations
real(dk)    ::  Upot,dUdr(1:3),dUdt  ! Perturbing potential and its derivatives
real(dk)    ::  f(1:3),p(1:3)        ! Total and non-conservative perturbing accelerations

! ==============================================================================

! INDEX OF STATE VECTORS

! z(1) = zeta1
! z(2) = zeta2
! z(3) = zeta3
! z(4) = zeta4
! z(5) = zeta5
! z(6) = zeta6
! z(7) = zeta7
! z(8) = zeta8 (physical time,
!               constant time element,
!               linear time element,
!               depending on flag_time)


! Safety check
!if (any(z/=z)) then
!	write(*,*) 'NaNs detected in the RHS, stopping execution.'
!	stop
!end if

! ==============================================================================
! 01. AUXILIARY QUANTITIES (1)
! ==============================================================================

! Store trig functions
sph = sin(phi)
cph = cos(phi)

rho   = 1._dk - z(1)*cph - z(2)*sph
rmag  = z(3)*rho
zeta  = z(1)*sph - z(2)*cph
emme  = sqrt(1._dk - z(1)**2 - z(2)**2)

cnu = (cph - z(1) + (zeta*z(2))/(emme + 1._dk))/rho
snu = (sph - z(2) - (zeta*z(1))/(emme + 1._dk))/rho

! ==============================================================================
! 02. POSITION IN INERTIAL FRAME, TIME
! ==============================================================================

! Intermediate frame unit vectors
x_vec = 2._dk*[ .5_dk - z(5)**2 - z(6)**2,  &
           &  z(4)*z(5) + z(6)*z(7),  &
           &  z(4)*z(6) - z(5)*z(7)]
y_vec = 2._dk*[  z(4)*z(5) - z(6)*z(7),  &
           & .5_dk - z(4)**2 - z(6)**2,  &
           &  z(5)*z(6) + z(4)*z(7)]

! Position in inertial frame
rV = rmag*(x_vec*cnu + y_vec*snu)

flag_time = eqs - 2

! Get time
if ( flag_time == 0 ) then
    ! Physical time
    t = z(8)
elseif  ( flag_time == 1 ) then
    ! Constant time element
    t = z(8) - z(3)**1.5_dk*(zeta - phi)
elseif  ( flag_time == 2 ) then
    ! Linear time element
    t = z(8) - z(3)**1.5_dk*zeta
end if

! ==============================================================================
! 03. PERTURBING POTENTIAL
! ==============================================================================

Upot = 0._dk; dUdt = 0._dk; dUdr = 0._dk
call PERT_EJ2K(insgrav,0,0,0,0,0,rV,vV,rmag,t,dUdr,Upot,dUdt)
dUdr = INERT2ORB_EDROMO(dUdr,z,cnu,snu)

! ==============================================================================
! 04. VELOCITY IN THE INERTIAL FRAME
! ==============================================================================

i_vec = x_vec*cnu + y_vec*snu
j_vec = y_vec*cnu - x_vec*snu
v_rad = zeta/(sqrt(z(3))*rho)
v_tan = sqrt((1._dk - z(1)**2 - z(2)**2)/(z(3)*rho**2) - 2._dk*Upot)
vV = v_rad*i_vec + v_tan*j_vec
vsq = v_rad**2 + v_tan**2
vmag = sqrt(vsq)
cosg = v_rad/vmag; sing = v_tan/vmag

! ==============================================================================
! 05. PERTURBING ACCELERATIONS
! ==============================================================================

! Initializations
p = 0._dk; f = 0._dk
call PERT_EJ2K(0,isun,imoon,idrag,iF107,iSRP,rV,vV,rmag,t,p)
p = INERT2ORB_EDROMO(p,z,cnu,snu)

! ==============================================================================
! 06. COMPUTE AUXILIARY QUANTITIES (2)
! ==============================================================================

f = p + dUdr

enne = sqrt(emme**2 - 2._dk*z(3)*rho**2*Upot)

aux0 = (p(1)*zeta + p(2)*enne + dUdt*sqrt(z(3))*rho)
zdot(3) = 2._dk*z(3)**3*aux0
L3 = zdot(3)/(2._dk*z(3))

aux1 = ((2._dk*Upot - f(1)*rmag)*(2._dk - rho + emme)*rmag)/(emme*(emme+1._dk))
aux2 = (L3*zeta*(rho-emme))/(emme*(emme+1._dk))
wz   = (enne - emme)/rho + aux1 + aux2

! ==============================================================================
! 07. COMPUTE RIGHT-HAND SIDE
! ==============================================================================

! In-plane
aux3 = (f(1)*rmag - 2._dk*Upot)*rmag

zdot(1) =  aux3*sph + L3*((1._dk+rho)*cph - z(1))
zdot(2) = -aux3*cph + L3*((1._dk+rho)*sph - z(2))

! Out-of-plane
aux4 = (f(3)*rmag**2)/(2._dk*enne)

zdot(4) =  aux4*(z(7)*cnu - z(6)*snu) + 0.5_dk*wz*z(5)
zdot(5) =  aux4*(z(6)*cnu + z(7)*snu) - 0.5_dk*wz*z(4)
zdot(6) = -aux4*(z(5)*cnu - z(4)*snu) + 0.5_dk*wz*z(7)
zdot(7) = -aux4*(z(4)*cnu + z(5)*snu) - 0.5_dk*wz*z(6)

! Time / Time Element
if (flag_time == 0) then
    zdot(8) = sqrt(z(3))*rmag
else if (flag_time == 1) then   ! Constant Time Element
    zdot(8) = z(3)**1.5_dk * ( aux3 + (zeta - 1.5_dk*phi)*zdot(3)/z(3) )
else if (flag_time == 2) then   ! Linear Time Element
    zdot(8) = z(3)**1.5_dk * ( 1._dk + aux3 + 2._dk*L3*zeta )
end if

end subroutine EDROMO_RHS


subroutine EDROMO_EVT(neq,phi,z,ng,roots)
! Description:
!    Finds roots to stop the integration for the EDromo formulation.
! 
! Revisions:
!    190110: Add header comments. Add check for collision with the Moon.
! 
! ==============================================================================

! MODULES
use SUN_MOON,    only: EPHEM
use AUXILIARIES, only: MJD0,MJDnext,MJDf,DU,TU
use PHYS_CONST,  only: secsPerDay,RE,reentry_radius_nd,ReqM
use SETTINGS,    only: eqs,imoon,imcoll

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
real(dk)  ::  t         ! Current time [-]
real(dk)  ::  rho, rmag, dmag
real(dk)  ::  r_vec(1:3), dummy(1:3), rMoon(1:3), vMoon(1:3)

! ==============================================================================

roots = 1._dk

! Get time
flag_time = eqs - 2
t = EDROMO_TE2TIME(z,phi,flag_time)

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

rho   = 1._dk - z(1)*cos(phi) - z(2)*sin(phi)
rmag  = z(3)*rho

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
  call EDROMO2CART(phi,z,r_vec,dummy,posOnly=.true.)
  r_vec = r_vec*DU
  call EPHEM(2, 1._dk, 1._dk, t, rMoon, vMoon)
  dmag = sqrt(dot_product( (r_vec - rMoon), (r_vec - rMoon) ) )
  roots(4) = dmag - ReqM

end if


end subroutine EDROMO_EVT


function EDROMO_PHI0(R,V,pot,GM,DU,TU)
! Description:
!    Computes initial value of the EDromo fictitious time according to the
!    suggestion in [1].
!
! ==============================================================================

implicit none
! Arguments
real(dk),intent(in)  ::  R(1:3),V(1:3),pot,GM,DU,TU
real(dk)  :: EDROMO_PHI0
! Locals
real(dk)  ::  R_nd(1:3),V_nd(1:3),GM_nd,Rmag_nd
real(dk)  ::  rvdot
real(dk)  ::  pot_nd,totEn

R_nd = R/DU; V_nd = V/(DU*TU); GM_nd = GM/(DU**3*TU**2); pot_nd = pot/(DU*TU)**2
Rmag_nd = sqrt(dot_product(R_nd,R_nd))
rvdot = dot_product(R_nd,V_nd)
totEn = .5_dk*dot_product(V_nd,V_nd) - GM_nd/Rmag_nd - pot_nd
EDROMO_PHI0 = atan2(rvdot*sqrt(-2._dk*totEn), 1._dk + 2._dk*totEn*Rmag_nd)

end function EDROMO_PHI0


! ==============================================================================
! 02. TRANSFORMATIONS AND PROCESSING PROCEDURES
! ==============================================================================


function EDROMO_TE2TIME(z,phi,flag_time)
! Description:
!    Gets the value of physical time from the EDromo state vector "z" and
!    fictitious time "phi".
!
! ==============================================================================

! VARIABLES
implicit none
! Arguments IN
real(dk),intent(in)     ::  z(1:8),phi
integer,intent(in)  ::  flag_time
! Function definition
real(dk)                ::  EDROMO_TE2TIME

! ==============================================================================

if ( flag_time == 0 ) then
    ! Physical time
    EDROMO_TE2TIME = z(8)

else if  ( flag_time == 1 ) then
    ! Constant time element
    EDROMO_TE2TIME = z(8) - z(3)**1.5_dk*(z(1)*sin(phi) - z(2)*cos(phi) - phi)

else if  ( flag_time == 2 ) then
    ! Linear time element
    EDROMO_TE2TIME = z(8) - z(3)**1.5_dk*(z(1)*sin(phi) - z(2)*cos(phi))

end if

end function EDROMO_TE2TIME


subroutine CART2EDROMO(R,V,t0,DU,TU,z,phi,W,flag_time)
! Description:
!    Converts from Cartesian coordinates to EDromo elements. It requires the
!    current values of the fictitious time, perturbing potential and initial
!    time.
!
!===============================================================================

! VARIABLES
implicit none

! Arguments
integer,parameter       ::  neq = 8		        ! Number of elements of EDromo state vector
real(dk),intent(in)     ::  R(1:3),V(1:3)		! Dimensional position and velocity [km,km/s]
real(dk),intent(in)     ::  DU,TU               ! Ref. quantities for non-dimensionalization
real(dk),intent(in)     ::  phi                 ! Initial phi value
real(dk),intent(in)     ::  t0               	! Initial time value [s]
real(dk),intent(in)     ::  W                 ! Potential energy [km^2/s^2]
integer,intent(in)      ::  flag_time		    ! = 0 physical time
         								        ! = 1 constant time element
								                ! = 2 linear time element
real(dk),intent(out)  ::  z(1:neq)              ! EDromo state vector

! Local variables
real(dk)  ::  y(1:6)           		            ! Cartesian state vector, ND
real(dk)  ::  Rmag,Vmag      	                ! Radius and velocity magnitudes, ND
real(dk)  ::  rV                                ! r dot v, ND
real(dk)  ::  h0(1:3),hmag0    	                ! Initial angular momentum and magnitude
real(dk)  ::  pot0           		            ! Potential, ND
real(dk)  ::  totEn0           		            ! Initial total energy, ND
real(dk)  ::  c0                                ! Generalized angular momentum
real(dk)  ::  cph,sph
real(dk)  ::  nu0,cnu,snu          	            ! Auxiliary variables
real(dk)  ::  i_vec(1:3),j_vec(1:3),k_vec(1:3)  ! Orbital frame unit vectors in inertial RF
real(dk)  ::  x_vec(1:3),y_vec(1:3)             ! Intermediate frame unit vectors in inertial RF
real(dk)  ::  aux                	            ! Auxiliary variables
real(dk)  ::  zero             		            ! Reference machine zero

! ==============================================================================

! ==============================================================================
! 01. NON-DIMENSIONALIZATION
! ==============================================================================
y(1:3) = R/DU
y(4:6) = V/(DU*TU)
pot0   = W/(DU*TU)**2

! Compute machine zero for comparisons
zero = epsilon(0._dk)

! ==============================================================================
! 02. IN-PLANE ELEMENTS
! ==============================================================================
Rmag = sqrt(dot_product(y(1:3),y(1:3)))
Vmag = sqrt(dot_product(y(4:6),y(4:6)))

cph = cos(phi)
sph = sin(phi)

! Total energy
totEn0  = .5_dk*Vmag**2 - 1._dk/Rmag + pot0

! Angular momentum
h0 = DCROSS_PRODUCT(y(1:3),y(4:6))
hmag0 = sqrt(dot_product(h0,h0))

! Generalized angular momentum
c0 = sqrt(hmag0**2 + 2._dk*Rmag**2*pot0)

! Dot product
rV = dot_product(y(1:3),y(4:6))

! IN-PLANE ELEMENTS
z(1) = (1._dk + 2._dk*totEn0*Rmag)*cph + rV*sqrt(-2._dk*totEn0)*sph
z(2) = (1._dk + 2._dk*totEn0*Rmag)*sph - rV*sqrt(-2._dk*totEn0)*cph
z(3) = -1._dk/(2._dk*totEn0)

! ==============================================================================
! 03. QUATERNION ELEMENTS
! ==============================================================================

nu0 = phi + 2._dk*atan2(rV,(c0 + Rmag*sqrt(-2._dk*totEn0)))
cnu = cos(nu0); snu = sin(nu0)

! Orbital frame unit vectors in IRF
i_vec = y(1:3)/Rmag
k_vec = h0/hmag0
j_vec = DCROSS_PRODUCT(k_vec,i_vec)

! Intermediate frame unit vectors in IRF
x_vec = i_vec*cnu - j_vec*snu
y_vec = j_vec*cnu + i_vec*snu

! SAFE INITIALIZATION OF THE QUATERNION
! The arguments of the roots have to be always positive, therefore we can safely use ABS()
aux = abs(1._dk + x_vec(1) + y_vec(2) + k_vec(3))
z(7) = .5_dk*sqrt(aux)
! Check for singularities and NaNs
if ( aux <= zero ) then
    aux    = abs((.5_dk*(k_vec(3) + 1._dk)))
    z(6)  = sqrt(aux)
    if ( aux <= zero ) then
        aux   = abs(.5_dk*(1._dk - y_vec(2)))
        z(4) = sqrt(aux)
        if ( aux <= zero ) then
            z(5) = 1._dk
        else
            z(5) = y_vec(1)/(2._dk*z(4))
        end if
    else
        z(4) = k_vec(1)/(2.*z(6))
        z(5) = k_vec(2)/(2.*z(6))
    end if
else
    z(4) = (y_vec(3) - k_vec(2))/(4._dk*z(7))
    z(5) = (k_vec(1) - x_vec(3))/(4._dk*z(7))
    z(6) = (x_vec(2) - y_vec(1))/(4._dk*z(7))
end if

! ==============================================================================
! 04. TIME / TIME ELEMENT
! ==============================================================================

if ( flag_time == 0 ) then
    ! Physical time
    z(8) = t0*TU
elseif  ( flag_time == 1 ) then
    ! Constant time element
    z(8) = t0*TU + z(3)**1.5*(z(1)*sph - z(2)*cph - phi)
elseif  ( flag_time == 2 ) then
    ! Linear time element
    z(8) = t0*TU + z(3)**1.5*(z(1)*sph - z(2)*cph)
end if

contains

FUNCTION DCROSS_PRODUCT(a,b)

! VARIABLES
implicit none
! Function definition
real(dk)        :: DCROSS_PRODUCT(1:3)
! Arguments
real(dk)        :: a(1:3),b(1:3)

DCROSS_PRODUCT(1) = a(2)*b(3) - a(3)*b(2)
DCROSS_PRODUCT(2) = a(3)*b(1) - a(1)*b(3)
DCROSS_PRODUCT(3) = a(1)*b(2) - a(2)*b(1)
END FUNCTION DCROSS_PRODUCT

end subroutine CART2EDROMO




subroutine EDROMO2CART(phi,z,r_vec,v_vec,posOnly)
! Description:
!    Transforms from the EDromo state vector "z", fictitious time "phi" and
!    potential "Upot" to Cartesian position and velocity "r_vec", "v_vec".
!    **All quantities are dimensionless, unlike in CART2EDROMO**.
! 
! Revisions:
!    190113: Add optional flag for the calculation of the position only.
!
! ==============================================================================

! MODULES
use NSGRAV,        only: PINES_NSG
use SETTINGS,      only: insgrav,eqs
use PHYS_CONST,    only: GE_nd,RE_nd
! VARIABLES
implicit none
! Arguments IN
real(dk),intent(in)   ::  z(1:8),phi
logical, optional, intent(in)  ::  posOnly
! Arguments OUT
real(dk),intent(out)  ::  r_vec(1:3),v_vec(1:3)

! Auxiliaries
real(dk)    ::  Upot
real(dk)    ::  sph,cph
real(dk)    ::  rho,rmag,zeta,emme
real(dk)    ::  cnu,snu
real(dk)    ::  x_vec(1:3),y_vec(1:3)
real(dk)    ::  i_vec(1:3),j_vec(1:3)
real(dk)    ::  v_rad,v_tan
real(dk)    ::  t
integer     ::  flag_time

! ==============================================================================

! ==============================================================================
! 01. COMPUTE AUXILIARY QUANTITIES
! ==============================================================================

! Store trig functions
sph = sin(phi)
cph = cos(phi)

rho   = 1._dk - z(1)*cph - z(2)*sph
rmag  = z(3)*rho
zeta  = z(1)*sph - z(2)*cph
emme  = sqrt(1._dk - z(1)**2 - z(2)**2)

cnu = (cph - z(1) + (zeta*z(2))/(emme + 1._dk))/rho
snu = (sph - z(2) - (zeta*z(1))/(emme + 1._dk))/rho

! ==============================================================================
! 02. COMPUTE POSITION IN INERTIAL FRAME
! ==============================================================================

! Intermediate frame unit vectors
x_vec = 2._dk*[ .5_dk - z(5)**2 - z(6)**2,  &
           &  z(4)*z(5) + z(6)*z(7),  &
           &  z(4)*z(6) - z(5)*z(7)]
y_vec = 2._dk*[  z(4)*z(5) - z(6)*z(7),  &
           & .5_dk - z(4)**2 - z(6)**2,  &
           &  z(5)*z(6) + z(4)*z(7)]

! Position in inertial frame
r_vec = rmag*(x_vec*cnu + y_vec*snu)

! ==============================================================================
! 03. PERTURBING POTENTIAL
! ==============================================================================

! By default, compute velocity
if(.not.(present(posOnly))) then
    flag_time = eqs - 2
    t = EDROMO_TE2TIME(z,phi,flag_time)
    Upot = 0._dk
    if (insgrav == 1) then
    call PINES_NSG(GE_nd,RE_nd,r_vec,t,pot=Upot)

    end if

! ==============================================================================
! 04. COMPUTE VELOCITY IN INERTIAL FRAME
! ==============================================================================

    ! Radial and tangential unit vectors in the inertial frame
    i_vec = x_vec*cnu + y_vec*snu
    j_vec = y_vec*cnu - x_vec*snu

    ! Radial and tangential components
    v_rad = zeta/(sqrt(z(3))*rho)
    v_tan = sqrt((1._dk - z(1)**2 - z(2)**2)/(z(3)*rho**2) - 2._dk*Upot)

    ! Velocity in the inertial frame
    v_vec = v_rad*i_vec + v_tan*j_vec

end if

end subroutine EDROMO2CART




function INERT2ORB_EDROMO(vI,z,cnu,snu)
! Description:
!    Transforms a vector vI from inertial to orbital axes through a rotation
!    matrix obtained fromo EDromo elements.
!
! ==============================================================================

! VARIABLES
implicit none

! Arguments
real(dk),intent(in)  ::  vI(1:3)
real(dk),intent(in)  ::  z(:)
real(dk),intent(in)  ::  cnu,snu
real(dk)             ::  INERT2ORB_EDROMO(1:3)
! Locals
real(dk)             ::  R1(1:3,1:3)    ! Inertial -> Intermediate rotation matrix
real(dk)             ::  R2(1:3,1:3)    ! Intermediate -> Orbital rotation matrix
real(dk)             ::  RTOT(1:3,1:3)  ! Inertial -> Orbital rotation matrix (= R2*R1)

! ==============================================================================

R1 = 2._dk*reshape([ .5_dk - z(5)**2 - z(6)**2, z(4)*z(5) + z(6)*z(7), z(4)*z(6) - z(5)*z(7),&
            &  z(4)*z(5) - z(6)*z(7), .5_dk - z(4)**2 -z(6)**2, z(5)*z(6) + z(4)*z(7), &
            &  z(4)*z(6) + z(5)*z(7), z(5)*z(6) - z(4)*z(7), .5_dk - z(4)**2 - z(5)**2 ], [3,3])
R1 = transpose(R1)
R2   = reshape([ cnu, -snu, 0._dk,&
              &   snu,  cnu, 0._dk,&
              &   0._dk,    0._dk, 1._dk ],[3,3])
RTOT = matmul(R2,R1)

INERT2ORB_EDROMO = matmul(RTOT,vI)

end function INERT2ORB_EDROMO




end module EDROMO
