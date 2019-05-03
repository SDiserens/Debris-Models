module INITIALIZE
! Description:
!    Contains wrapper initialization procedures for Thalassa.
!
! Author:
!    Davide Amato
!    Space Dynamics Group - Technical University of Madrid
!    davideamato@email.arizona.edu
!
! ==============================================================================

use KINDS, only: dk
implicit none


contains


subroutine INIT_STATE(eqs,R0,V0,MJD0,neq,y0,x0)
! Description:
!    Initializes the state vector "y0" and the independent variable "x0" from
!    Cartesian coordinates and initial epoch.
!
! ==============================================================================

! MODULES
use AUXILIARIES,   only: DU,TU
use PHYS_CONST,    only: GE,RE
use SETTINGS,      only: insgrav
use EDROMO,        only: EDROMO_PHI0,CART2EDROMO
use KUST_STI,      only: CART2KS
use STI_SCHE,      only: CART2STISCHE
use NSGRAV,        only: PINES_NSG

! VARIABLES
implicit none
! Arguments
integer,intent(in)               ::  eqs
real(dk),intent(in)              ::  R0(1:3),V0(1:3),MJD0
integer,intent(out)              ::  neq
real(dk),intent(out),allocatable ::  y0(:)
real(dk),intent(out)             ::  x0
! Locals
real(dk)                         ::  Rm
real(dk)                         ::  U0     ! Perturbing potential [km^2/s^2]
real(dk)                         ::  F0(1:3)
integer                          ::  ftime  ! Flag for time-like variable

! ==============================================================================

! Deallocate x0 if already allocated for some reason
if (allocated(y0)) deallocate(y0)

select case (eqs)

    case(1)   ! Cowell, 1st order
        neq = 6
        allocate(y0(1:neq))
        y0(1:3) = R0/DU
        y0(4:6) = V0/(DU*TU)
        ! Set initial value of physical time to 0 always, as to reduce round-off
        ! problems.
        x0 = 0._dk

    case(2:4) ! EDromo
        neq = 8
        ftime = eqs - 2
        allocate(y0(1:neq))
        Rm = sqrt(dot_product(R0,R0))
        U0 = 0._dk
        if (insgrav == 1) then
          call PINES_NSG(GE,RE,R0,0._dk,pot=U0)

        end if
        x0 = EDROMO_PHI0(R0,V0,U0,GE,DU,TU)

        ! As before, set initial physical time to 0. The actual MJD will be
        ! recovered later.
        call CART2EDROMO(R0,V0,0._dk,DU,TU,y0,x0,U0,ftime)
    
    case(5:6) ! KS
        neq = 10
        ftime = eqs - 5
        allocate(y0(1:neq))
        Rm = sqrt(dot_product(R0,R0))
        U0 = 0._dk
        if (insgrav == 1) then
          call PINES_NSG(GE,RE,R0,0._dk,pot=U0)

        end if
        call CART2KS(R0,V0,0._dk,GE,DU,TU,y0,U0,ftime)
        x0 = EDROMO_PHI0(R0,V0,U0,GE,DU,TU)
    
    case(7:8) ! Stiefel-Scheifele
        neq = 10
        ftime = eqs - 7
        allocate(y0(1:neq))
        Rm = sqrt(dot_product(R0,R0))
        U0 = 0._dk
        if (insgrav == 1) then
          call PINES_NSG(GE,RE,R0,0._dk,pot=U0)

        end if
        x0 = EDROMO_PHI0(R0,V0,U0,GE,DU,TU)
        call CART2STISCHE(R0,V0,0._dk,GE,DU,TU,y0,x0,U0,ftime)
        
end select

end subroutine INIT_STATE


end module INITIALIZE
