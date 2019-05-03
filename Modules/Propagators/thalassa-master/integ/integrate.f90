module INTEGRATE
! Description:
!    Contains wrapper subroutines for the solvers employed in Thalassa.
!
! Author:
!    Davide Amato
!    Space Dynamics Group - Technical University of Madrid
!    davideamato@email.arizona.edu
!
! ==============================================================================

use KINDS, only: dk
implicit none
abstract interface
  subroutine FTYPE1(neq,t,y,ydot)
    import  ::  dk
    implicit none
    integer,intent(in)    ::  neq
    real(dk),intent(in)       ::  t
    real(dk),intent(in)       ::  y(1:neq)
    real(dk),intent(out)      ::  ydot(1:neq)
  end subroutine FTYPE1

  subroutine FTYPE2(neq,t,y,ydot,yddot)
    import  ::  dk
    implicit none
    integer,intent(in)    ::  neq
    real(dk),intent(in)       ::  t
    real(dk),intent(in)       ::  y(1:neq)
    real(dk),intent(in)       ::  ydot(1:neq)
    real(dk),intent(out)      ::  yddot(1:neq)
  end subroutine FTYPE2

  subroutine EVENTS(neq,t,y,ng,roots)
    import  ::  dk
    implicit none
    integer,intent(in)  ::  neq,ng
    real(dk),intent(in)     ::  t,y(1:neq)
    real(dk),intent(out)    ::  roots(1:ng)
  end subroutine EVENTS

end interface


contains


subroutine INTSTEP(EOM,EVT,integ,eqs,neq,xi,yi,dx,xf,yf,rtol,atol,isett,lrw,&
&rwork,liw,iwork)
! Description:
!    Performs one integration step with the integrator specified by "integ".* It
!    advances the state vector yi(xi) to yf(xf), where xf = xi + dx, doing so by
!    integrating the equations of motion specified in the subroutine EOM. The
!    user can provide an event function through the subroutine EVT.
!    
!    * At the present moment, only the LSODAR integrator is available.
!      (DA, 19 Feb 2018)
!
! ==============================================================================

! VARIABLES
implicit none
external SLSODAR,DLSODAR
! Arguments
integer,intent(in)      ::  integ                 ! Integrator flag
integer,intent(in)      ::  eqs                   ! Type of equations
integer,intent(in)      ::  neq                   ! Number of equations
integer,intent(in)      ::  lrw,liw               ! Length of work arrays
real(dk),intent(in)     ::  yi(1:neq),xi          ! State vector and ind. var. at beginning of step
real(dk),intent(in)     ::  dx                    ! Independent variable
real(dk),intent(in)     ::  rtol(:),atol(:)             ! Absolute and relative tolerances
integer,intent(inout)   ::  isett(:),iwork(1:liw) ! Settings and integer work arrays
real(dk),intent(inout)  ::  rwork(1:lrw)          ! Real work array
real(dk),intent(out)    ::  yf(1:neq),xf          ! State vector and ind. variables at end of step
procedure(FTYPE1)       ::  EOM                   ! Equations of motion
procedure(EVENTS)       ::  EVT                   ! Event function

! Locals
real(dk)  ::  y(1:neq),x
! LSODAR settings
integer   ::  itol,itask,istate,iopt,jt,nevts
integer   ::  jroot(1:10)


! ==============================================================================

! Array "isett" is an integer array containing options for the solver that is
! currently being used. Check the following switch construct for the meaning of
! the elements of isett.

select case (integ)
    case(1) ! LSODAR

    ! Unpack integration options
    itol   = isett(1)
    itask  = isett(2)
    istate = isett(3)
    iopt   = isett(4)
    jt     = isett(5)
    nevts  = isett(6)
    jroot  = isett(7:16)

    y = yi; x = xi; xf = xi + dx
    rwork(5) = rwork(5)*(dx/abs(dx))
    
    ! Switch for quad or double precision
    if (dk == 8) then

        call SLSODAR(EOM,neq,y,x,xf,itol,rtol,atol,itask,istate,iopt,rwork,&
        &lrw,iwork,liw,FAKE,jt,EVT,nevts,jroot)

    else if (dk == 16) then

        call DLSODAR(EOM,neq,y,x,xf,itol,rtol,atol,itask,istate,iopt,rwork,&
        &lrw,iwork,liw,FAKE,jt,EVT,nevts,jroot)

    end if

    ! Re-pack integration options
    isett(1) = itol
    isett(2) = itask
    isett(3) = istate
    isett(4) = iopt
    isett(5) = jt
    isett(6) = nevts
    isett(7:16) = jroot
    yf = y
    xf = x

end select


end subroutine INTSTEP


subroutine SET_SOLV(integ,eqs,neq,tol,isett,iwork,rwork,rtols,atols)
! Description:
!    Sets solver settings.
! 
! Author:
!    Davide Amato
!    The University of Arizona
!    davideamato@email.arizona.edu
!
! Revisions:
!    180531: Set LSODAR unit to log file.
!    
! ==============================================================================

! MODULES
use PHYS_CONST, only: twopi
use IO,         only: id_log

! VARIABLES
implicit none
! Arguments
integer,intent(in)   ::  integ,eqs,neq
real(dk),intent(in)  ::  tol
integer,intent(inout)   ::  isett(:),iwork(:)
real(dk),intent(inout)  ::  rwork(:)
real(dk),allocatable,intent(out)  ::  rtols(:),atols(:)
! Locals
real(dk)    ::  liw,lrw

external XSETUN

! ==============================================================================

isett = 0; iwork = 0
rwork = 0._dk

! Set default values for solver options
select case (integ)
    case(1) ! SLSODAR,DLSODAR

      ! Default values for isett
      isett(1) = 4      ! itol
      isett(2) = 1      ! itask
      isett(3) = 1      ! istate
      isett(4) = 1      ! iopt
      isett(5) = 2      ! jt
      isett(6) = 4      ! nevts (max. = 10, see length of jroot)
      isett(7:16) = 0   ! jroot

      ! Default values for iwork.
      ! Next is max. number of integration steps per output steps. Increase this
      ! value for very large output steps or very small tolerances.
      iwork(6) = 10000000
      ! Next is the user-assigned initial step size. Set to 0 to let LSODAR
      ! estimate it.

      ! Allocate tolerances
      if (allocated(rtols)) deallocate(rtols); allocate(rtols(1:neq))
      if (allocated(atols)) deallocate(atols); allocate(atols(1:neq))
      rtols = tol
      atols = tol

      select case (eqs)
          case(1,-1)
              ! For Cowell, let the solver estimate the step size.
              rwork(5) = 0._dk

          case(2:8)
              ! For regularized formulations using angle-like fictitious times,
              ! estimate the step size as ~100th of a period, and scale according
              ! to tolerance.
              rwork(5) = -0.01_dk*twopi/log10(tol)

              ! For EDromo(c), set only absolute tolerance for the time element.
              if (eqs == 3) rtols(8) = 1._dk

      end select
      
      ! Send error messages to log file
      call XSETUN(id_log)

end select

end subroutine SET_SOLV


function SET_DX(eqs,tstep,TU)
! Description:
!    Set the step size in independent variable depending on the formulation
!    being used.
!
! ==============================================================================

! MODULES
use PHYS_CONST, only: secsPerDay
! VARIABLES
implicit none
integer,intent(in)   ::  eqs
real(dk),intent(in)  ::  tstep,TU
real(dk)  ::  SET_DX

! ==============================================================================

select case (eqs)
    case (1) ! Cowell
        ! For Cowell, the step size is simply the step size in days, non-dimensionalized.
        SET_DX = tstep*secsPerDay*TU

    case (2:8) ! EDromo, KS, Stiefel-Scheifele
        ! For regularized formulations, set the step size to +inf since they are
        ! to be stopped by event location.
        SET_DX = (tstep/abs(tstep))*huge(0._dk)

end select

end function SET_DX


subroutine FAKE; end subroutine FAKE


end module INTEGRATE
