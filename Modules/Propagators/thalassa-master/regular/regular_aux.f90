module REGULAR_AUX
! Description:
!    Contains auxiliary procedures necessary for the integration of regularized
!    formulations.
!
! Author:
!    Davide Amato
!    Space Dynamics Group - Technical University of Madrid
!    davideamato@email.arizona.edu
!
! ==============================================================================

! MODULES
use KINDS, only: dk
use EDROMO
use KUST_STI
use STI_SCHE

! VARIABLES
implicit none


contains


function PHYSICAL_TIME(eqs,neq,x,y)
! Description:
!    Computes the value of the dimensionless physical time, given the values of
!    the state vector and of the independent variable.
!
! ==============================================================================

! VARIABLES
implicit none
integer,intent(in)   ::  eqs,neq
real(dk),intent(in)  ::  y(1:neq),x
real(dk)             ::  PHYSICAL_TIME
! Locals
integer              ::  ftime

! ==============================================================================

select case (eqs)
    case(1,-1) ! Cowell
        PHYSICAL_TIME = x

    case(2:4)  ! EDromo
        ftime = eqs - 2
        PHYSICAL_TIME = EDROMO_TE2TIME(y,x,ftime)
    
    case(5:6)  ! KS
        ftime = eqs - 5
        PHYSICAL_TIME = KS_TE2TIME(y,ftime)
    
    case(7:8)
        ftime = eqs - 7
        PHYSICAL_TIME = STISCHE_TE2TIME(y,x,ftime)

end select

end function PHYSICAL_TIME


function CARTESIAN(eqs,neq,DU,TU,x,y,ydot)
! Description:
!    Computes the Cartesian coordinates in the inertial reference frame from
!    the state vector and independent variable. Output is dimensional.
!
! ==============================================================================

! VARIABLES
implicit none
! Arguments
integer,intent(in)            ::  eqs,neq
real(dk),intent(in)           ::  DU,TU,x,y(1:neq)
real(dk),intent(in),optional  ::  ydot(1:neq)
real(dk)                      ::  CARTESIAN(1:6)
! Locals
real(dk)    ::  R(1:3),V(1:3)

! ==============================================================================

select case (eqs)
    case(-1)  ! Cowell, 2nd order
        CARTESIAN = [y*DU,ydot*DU*TU]

    case(1)   ! Cowell, 1st order
        CARTESIAN = [y(1:3)*DU,y(4:6)*DU*TU]

    case(2:4) ! EDromo
        call EDROMO2CART(x,y,R,V)
        CARTESIAN = [R*DU,V*DU*TU]
    
    case(5:6) ! KS
        call KS2CART(y,R,V)
        CARTESIAN = [R*DU,V*DU*TU]
    
    case(7:8) ! Stiefel-Scheifele
        call STISCHE2CART(x,y,R,V)
        CARTESIAN = [R*DU,V*DU*TU]

end select

end function CARTESIAN


end module REGULAR_AUX
