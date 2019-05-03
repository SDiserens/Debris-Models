module PROPAGATE
! Description:
!    Orbit propagation procedures for Thalassa. These are wrapper subroutines
!    that take care of all the main aspects related to the propagation, such
!    as:
!    - Initialization of the state vector,
!    - Initialization of integrator-related quantities,
!    - Integration loop
!    - Online and offline processing.
!
! Author:
!    Davide Amato
!    The University of Arizona
!    Space Dynamics Group - Technical University of Madrid
!    davideamato@email.arizona.edu
! 
! Revisions:
!    180531: Add logging facilities, exit code.
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


subroutine DPROP_REGULAR(R0,V0,tspan,tstep,cart,int_steps,tot_calls,exitcode)
! Description:
!    Propagates an orbit for "tspan" days, starting from MJD0. The propagation
!    is performed using either regularized formulations or unregularized Cowell.
! 
! Author:
!    Davide Amato
!    The University of Arizona
!    davideamato@email.arizona.edu
!    
! Revisions:
!    180531: Add logging facilities, exit code.
!
! ==============================================================================

! MODULES
use AUXILIARIES, only: SET_UNITS
use INITIALIZE,  only: INIT_STATE
use INTEGRATE,   only: SET_SOLV,SET_DX
use COWELL,      only: COWELL_RHS,COWELL_EVT
use EDROMO,      only: EDROMO_RHS,EDROMO_EVT
use KUST_STI,    only: KS_RHS,KS_EVT
use STI_SCHE,    only: STISCHE_RHS,STISCHE_EVT
use REGULAR_AUX, only: PHYSICAL_TIME,CARTESIAN
use AUXILIARIES, only: MJD0,MJDnext,MJDf,DU,TU
use IO,          only: EXIT_MSG
use IO,          only: id_log
use PHYS_CONST,  only: GE,secsPerDay,GE,RE,GE_nd,RE_nd,ERR_constant,&
&ERR_constant_nd,pi,reentry_height,reentry_radius_nd
use SETTINGS,    only: eqs,tol

! VARIABLES
implicit none
! ARGUMENTS
real(dk),intent(in)  ::  R0(1:3),V0(1:3)
real(dk),intent(in)  ::  tspan,tstep
real(dk),intent(out),allocatable  ::  cart(:,:)
integer,intent(out)   ::  int_steps,tot_calls,exitcode
! LOCALS
integer   ::  neq                  ! N. of equations
real(dk)  ::  x0                   ! Initial value of indep. variable
real(dk)  ::  dx                   ! Step size in independent variable
real(dk),allocatable  ::  y0(:)    ! Initial value of state vector
! Integration settings
integer,parameter  ::  liw=500,lrw=500   ! Length of work arrays, increase if needed.
integer   ::  iwork(1:liw)               ! Integer work array
real(dk)  ::  rwork(1:lrw)               ! Real work array
integer   ::  isett(1:20)                ! Solver option flags. Increase size if needed.

! Results
integer               ::  ip
integer               ::  npts,nels
real(dk),allocatable  ::  yx(:,:)
real(dk)              ::  t

! LSODAR - individual tolerances
real(dk),allocatable  ::  atols(:)
real(dk),allocatable  ::  rtols(:)

! ==============================================================================

! ==============================================================================
! 01. INITIALIZATIONS
! ==============================================================================

! Set reference units and non-dimensionalizations
call SET_UNITS(R0,GE)
GE_nd = GE/(DU**3*TU**2)
RE_nd = RE/DU
ERR_constant_nd = 2._dk*pi*ERR_constant/secsPerDay/TU
reentry_radius_nd = (reentry_height + RE)/DU

! Set times (TBD)
MJDf = MJD0 + tspan
MJDnext = MJD0 + tstep

! Initialize state vector and independent variable
call INIT_STATE(eqs,R0,V0,MJD0,neq,y0,x0)

write(id_log,'(a,i2)') 'Initialized the state vector for eqs = ',eqs

! ==============================================================================
! 02. INTEGRATION LOOP
! ==============================================================================

! Solver initialization
call SET_SOLV(1,eqs,neq,tol,isett,iwork,rwork,rtols,atols)

dx = SET_DX(eqs,tstep,TU)

write(id_log,'(a)') 'Initialized solver settings, entering the integration loop.'

! Choose equations of motion and start MAIN INTEGRATION LOOP
select case (eqs)
    case(1)   ! Cowell, 1st order
        call INTLOOP(COWELL_RHS,COWELL_EVT,1,eqs,neq,y0,x0,dx,tstep,yx,rtols,&
        &atols,isett,liw,iwork,lrw,rwork,exitcode)

    case(2:4) ! EDromo
        call INTLOOP(EDROMO_RHS,EDROMO_EVT,1,eqs,neq,y0,x0,dx,tstep,yx,rtols,&
        &atols,isett,liw,iwork,lrw,rwork,exitcode)
    
    case(5:6) ! KS
        call INTLOOP(KS_RHS,KS_EVT,1,eqs,neq,y0,x0,dx,tstep,yx,rtols,&
        &atols,isett,liw,iwork,lrw,rwork,exitcode)
    
    case(7:8) ! Stiefel-Scheifele
        call INTLOOP(STISCHE_RHS,STISCHE_EVT,1,eqs,neq,y0,x0,dx,tstep,yx,rtols,&
        &atols,isett,liw,iwork,lrw,rwork,exitcode)
    
end select

call EXIT_MSG(exitcode)

! ==============================================================================
! 03. OFFLINE PROCESSING
! ==============================================================================

npts = size(yx,1)
nels = size(yx,2)
if (allocated(cart)) deallocate(cart)
allocate(cart(1:npts,1:7))

do ip=1,npts
    t = PHYSICAL_TIME(eqs,neq,yx(ip,1),yx(ip,2:nels))
    cart(ip,1)   = MJD0 + t/TU/secsPerDay
    cart(ip,2:7) = CARTESIAN(eqs,neq,DU,TU,yx(ip,1),yx(ip,2:nels))

end do

! Save total number of function calls and number of steps taken
int_steps = iwork(11)
tot_calls = iwork(12)

end subroutine DPROP_REGULAR


subroutine INTLOOP(EOM,EVT,integ,eqs,neq,y0,x0,dx,tstep,yx,rtol,atol,isett,liw,&
&iwork,lrw,rwork,exitcode)
! Description:
!    Performs the integration loop for the equations of motion specified through
!    the subroutine EOM. The stop condition is detected through event location.
!    The user also supplies the event function EVT.
! 
! Author:
!    Davide Amato
!    The University of Arizona
!    davideamato@email.arizona.edu
!    
! Revisions:
!    180531: Add logging facilities, exit code.
!
! ==============================================================================

! MODULES
use AUXILIARIES, only: MJDnext,MJD0,MJDf
use INTEGRATE,   only: INTSTEP
use SETTINGS,    only: mxstep,verb

! VARIABLES
implicit none
! Arguments
integer,intent(in)      ::  integ,eqs          ! Integrator flag and type of equations
integer,intent(in)      ::  neq                ! Number of equations
integer,intent(in)      ::  liw,lrw            ! Length of work arrays
real(dk),intent(in)     ::  y0(1:neq),x0,dx    ! Initial values and step size in independent var.
real(dk),intent(in)     ::  tstep              ! Step size in phys. time (days)
real(dk),intent(in)     ::  rtol(:),atol(:)    ! Integration tolerances
integer,intent(inout)   ::  isett(:)           ! Integrator settings
integer,intent(inout)   ::  iwork(1:liw)       ! Integer work array
real(dk),intent(inout)  ::  rwork(1:lrw)       ! Real work array
real(dk),intent(out),allocatable  ::  yx(:,:)  ! Output trajectory
integer,intent(out)     ::  exitcode           ! Exit code for diagnostics
procedure(FTYPE1)    ::  EOM
procedure(EVENTS)    ::  EVT
! Locals
real(dk)             ::  yprev(1:neq),xprev,ycur(1:neq),xcur
real(dk)             ::  auxy(0:mxstep,1:11)
integer              ::  iint,iwr
logical              ::  quit_flag
! Diagnostics
integer              ::  nsteps,print_each,i_print

! ==============================================================================

! Initializations
yprev = y0; xprev = x0
auxy = 0._dk
iint = 1
auxy(1,1:neq+1) = [x0,y0]
quit_flag = .false.
exitcode  = -10

! Approximate number of steps to be taken
nsteps = int((MJDf - MJD0)/(MJDnext - MJD0))
print_each = nsteps/20
i_print = 1

! MAIN LOOP
do

    ! Print message every print_each steps
    if ((i_print - print_each == 0) .and. (verb == 1)) then
      write(*,'(a,f9.2,a)') 'Progress: ',real(iint)/real(nsteps)*100.,'%'
      i_print = 0
    end if

    call INTSTEP(EOM,EVT,integ,eqs,neq,xprev,yprev,dx,xcur,ycur,rtol,atol,isett,&
    &lrw,rwork,liw,iwork)

    ! Save to output
    auxy(iint+1,1:neq+1) = [xcur,ycur]

    ! Exit conditions
    quit_flag = QUIT_LOOP(eqs,neq,integ,isett,exitcode,xcur,ycur)
    if (quit_flag) then
        exit

    else if (iint == mxstep) then
        write(*,*) 'WARNING: Maximum number of steps reached.'
        write(*,*) 'mxstep = ',mxstep
        write(*,*) 'xcur   = ',xcur

        exitcode = -3
        exit
    
    else if (any(ycur /= ycur)) then
        write(*,*) 'ERROR: NaNs detected in the state vector.'
        write(*,*) 'Try specifying another tolerance, or checking for'
        write(*,*) 'inconsistencies in your physical model.'
        write(*,*) 'Propagation is being STOPPED.'

        exitcode = -2
        exit

    end if

    ! Advance solution
    xprev = xcur
    yprev = ycur
    MJDnext = MJDnext + tstep
    iint = iint + 1
    i_print = i_print + 1

end do

! Dump output in yx
if (allocated(yx)) deallocate(yx)
allocate(yx(1:iint+1,1:neq+1))
do iwr=1,iint+1
  yx(iwr,1:neq+1) = auxy(iwr,1:neq+1)

end do

end subroutine INTLOOP


function QUIT_LOOP(eqs,neq,integ,isett,exitcode,x,y)
! Description:
!    Quits the integration loop when satisfying an exit condition.
! 
! Author:
!    Davide Amato
!    The University of Arizona
!    davideamato@email.arizona.edu
!    
! Revisions:
!    180531: Add logging facilities, exit code.
!    190113: Add check on Moon collision.
!
! ==============================================================================

! MODULES
use AUXILIARIES, only: MJD0,TU
use REGULAR_AUX, only: PHYSICAL_TIME
use PHYS_CONST,  only: mzero,secsPerDay,reentry_height
! VARIABLES
implicit none
! Arguments
integer,intent(in)     ::  neq,eqs,integ,isett(:)
real(dk),intent(in)    ::  y(:),x
integer,intent(inout)  ::  exitcode
logical                ::  QUIT_LOOP
! Locals
character(len=*),parameter  :: fmtstr = '(a, g10.3, 2(a, g14.7), a)'
integer   :: jroot(1:10)
real(dk)  :: tcur,MJDcur

! ==============================================================================

QUIT_LOOP = .false.
! The exit conditions are (for SLSODAR/DLSODAR):
!    jroot(2) = 1: reaching of final time.
!    jroot(3) = 1: Earth re-entry.
!    jroot(4) = 1: Moon collision.

select case (integ)
    case(1) ! SLSODAR, DLSODAR
        ! Unpack jroot
        jroot = isett(7:16)

        ! Nominal exit conditions.
        if (any(jroot(2:4) == 1)) then
            QUIT_LOOP = .true.
            exitcode = 0
            if (jroot(3) == 1 .or. jroot(4) == 1) then
              tcur = PHYSICAL_TIME(eqs,neq,x,y)
              MJDcur = MJD0 + tcur/TU/secsPerDay
              if (jroot(3) == 1) then
                  write(*,fmtstr) 'Earth re-entry detected, geoc. height <= ',&
                  &reentry_height,' km, MJD = ',MJDcur,' (UTC), duration = ',&
                  &tcur/TU/secsPerDay/365.25_dk,' years.'

              else if (jroot(4) == 1) then
                  write(*,fmtstr) 'Moon collision detected, selenoc. height <= ',&
                  &reentry_height,' km, MJD = ',MJDcur,' (UTC), duration = ',&
                  &tcur/TU/secsPerDay/365.25_dk,' years.'

              end if

              exitcode = 1

            end if

        end if

        ! Exceptions.
        ! For LSODAR, these are signalled by istate (= isett(3)) < 0.
        if (isett(3) < 0) then
            QUIT_LOOP = .true.

            exitcode = -1

        end if

end select

end function QUIT_LOOP

end module PROPAGATE
