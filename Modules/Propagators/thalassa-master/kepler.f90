module KEPLER
! Description:
!    Root-finding procedures for the Kepler equation.
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

function KESOLVE(ecc,M,tol)
! Solves the Kepler equation to the desired tolerance using the regula falsi
! method.

! VARIABLES
implicit none
! Arguments
real(dk),intent(in)   ::  ecc,M,tol
!logical,intent(out),optional   ::  flag
! Function def
real(dk)  ::  KESOLVE

! Locals
real(dk)  ::  zero
real(dk)  ::  sinM
real(dk)  ::  K0,K1,M1
integer   ::  it
real(dk)  ::  Enext,Knext,Ecurr,Kcurr,Eprev,Kprev

! ==============================================================================

!if (present(flag)) flag = .false.

zero = epsilon(0._dk)
sinM = sin(M)
if (abs(sinM) <= zero .or. ecc <= zero) then
    KESOLVE = M
    return
end if

! Plant seeds for the regula falsi
K0 = KEPEQ(ecc,M,M)
M1 = M + sinM/abs(sinM)*ecc
K1 = KEPEQ(ecc,M1,M)

Kprev = K0; Eprev = M
Kcurr = K1; Ecurr = M1

it = 0
do
    
    Enext = (Ecurr*Kprev - Eprev*Kcurr)/(Kprev - Kcurr)
    Knext = KEPEQ(ecc,Enext,M)
    
    ! Update Kprev, Eprev if the sign changed
    if (Knext*Kprev > 0._dk) then
        Eprev = Ecurr
        Kprev = Kcurr
    end if
    
    Ecurr = Enext; Kcurr = Knext
    
    it = it + 1
    
    if (abs(Kcurr) <= tol) then
        KESOLVE = Ecurr
        return
    end if
    
    if (it > 100) then
        KESOLVE = Ecurr
        write(*,*) 'KESOLVE did not converge.'
        write(*,*) 'Kcurr = ',Kcurr
!       if (present(flag)) flag = .true.
        return
    end if

end do

end function KESOLVE

function KEPEQ(ecc,E,M)

! VARIABLES
implicit none
real(dk),intent(in)  ::  ecc,E,M
real(dk)             ::  KEPEQ

KEPEQ = E - ecc*sin(E) - M

end function KEPEQ


end module KEPLER
