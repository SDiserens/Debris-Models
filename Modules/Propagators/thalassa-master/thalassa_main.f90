program THALASSA
! Description:
! Thalassa propagates Earth-bound orbits from an initial to a final epoch. It
! uses either Newtonian or regularized equations of motion written in the
! EMEJ2000 frame. It takes into account the following perturbations:
! - Non-spherical Earth gravitational potential
! - Third-body perturbations from Sun and Moon
! - Drag
! - SRP
!
! The choice of the mathematical model and constants reflect those used in
! STELA, a widely-known semi-analytical propagator currently in usage for studying
! the LEO, MEO and GEO regions.
! Also, the user can choose between the following numerical integrators:
! - LSODAR
!
! Author:
!    Davide Amato
!    Space Dynamics Group - Technical University of Madrid
!    The University of Arizona
!    davideamato@email.arizona.edu
! 
! Revisions:
!    180409: v0.9
!    180523: v0.9.1
!    180531: v0.9.2. Add output of log file.
!    180703: v1.0. Add time-dependent solar flux for J77 and MSIS-00. Consider
!            geodetic height for MSIS-00. Add batch scripts for large-scale
!            propagation.
!    180807: v1.1. Calculation of the gravitational potential now uses the Pines
!            method.
!    181006: v1.2. Add computation of truncated third-body acceleration.
!    190405: Only load SPICE kernels when iephem = 1.
!
! ==============================================================================


! MODULES
use KINDS,       only: dk
use AUXILIARIES, only: MJD0
use IO,          only: READ_IC,CREATE_OUT,DUMP_TRAJ,CREATE_LOG
use CART_COE,    only: COE2CART,CART2COE
use PHYS_CONST,  only: READ_PHYS,GMST_UNIFORM
use NSGRAV,      only: INITIALIZE_NSGRAV
use PROPAGATE,   only: DPROP_REGULAR
use SUN_MOON,    only: INITIALIZE_LEGENDRE
use SETTINGS,    only: READ_SETTINGS
use IO,          only: id_cart,id_orb,id_stat,id_log,object_path
use SETTINGS,    only: gdeg,gord,iephem,isun,imoon,outpath,tol,eqs,input_path
use PHYS_CONST,  only: GE,d2r,r2d,secsPerDay,secsPerSidDay,twopi
use SUN_MOON,    only: GslSun,GslMoon

implicit none

! VARIABLES
! Initial conditions (EMEJ2000)
real(dk)  ::  COE0(1:6),COE0_rad(1:6)
real(dk)  ::  R0(1:3),V0(1:3)
real(dk)  ::  GMST0,aGEO
real(dk)  ::  period
! Integration span and dt [solar days]
real(dk)  ::  tspan,tstep
! Trajectory
integer               ::  npts,ipt
real(dk),allocatable  ::  cart(:,:),orb(:,:)
real(dk)              ::  R(1:3),V(1:3)
real(dk)              ::  lifetime_yrs
! Measurement of CPU time, diagnostics
integer  ::  rate,tic,toc
real(dk) ::  cputime
integer  ::  exitcode
! Function calls and integration steps
integer  ::  int_steps,tot_calls
! Command arguments
integer  ::  command_arguments
! Date & time
character(len=8)   :: date_start, date_end
character(len=10)  :: time_start, time_end
character(len=5)   :: zone
! Paths
character(len=512) :: earth_path,phys_path

! ==============================================================================

! ==============================================================================
! 01. COMMAND LINE PARSING
! ==============================================================================

command_arguments = COMMAND_ARGUMENT_COUNT()
input_path  = './in/input.txt'
object_path = './in/object.txt'
earth_path = './data/earth_potential/GRIM5-S1.txt'
phys_path = './data/physical_constants.txt'
if (command_arguments > 0) call GET_COMMAND_ARGUMENT(1,input_path)
if (command_arguments > 1) call GET_COMMAND_ARGUMENT(2,object_path)
if (command_arguments > 2) call GET_COMMAND_ARGUMENT(3,earth_path)
if (command_arguments > 3) call GET_COMMAND_ARGUMENT(4,phys_path)

! ==============================================================================
! 02. INITIALIZATIONS
! ==============================================================================

! Start clock
call SYSTEM_CLOCK(tic,rate)

! Read initial conditions, settings and physical model data.
call READ_IC(MJD0,COE0)
call READ_SETTINGS(tspan,tstep)
call READ_PHYS(phys_path)

! Initialize Earth data
call INITIALIZE_NSGRAV(earth_path)

! Initialize Legendre coefficients, if needed
if (isun > 1) then
  call INITIALIZE_LEGENDRE(isun,GslSun)

end if
if (imoon > 1) then
  call INITIALIZE_LEGENDRE(imoon,GslMoon)
  
end if

! Initialize output (if not done by python script already)
if (command_arguments == 0) then
  call SYSTEM('mkdir -p '//trim(outpath))
  call SYSTEM('cp in/*.txt '//trim(outpath))
end if

! Log start of propagation
call CREATE_LOG(id_log,outpath)
call DATE_AND_TIME(date_start,time_start,zone)
write(id_log,'(a)') 'Start logging on '//date_start//'T'//time_start//' UTC'&
//zone//'.'
write(id_log,'(a)') 'Location of settings file: '//trim(input_path)
write(id_log,'(a)') 'Location of initial conditions file: '//trim(object_path)

! Load SPICE kernels - this is only necessary if we are using SPICE in this run.
if (iephem == 1) then
  call FURNSH('./data/kernels_to_load.furnsh')
  write(id_log,'(a)') 'SPICE kernels loaded through '//'./data/kernels_to_load.furnsh'

end if

! ==============================================================================
! 03. TEST PROPAGATION
! ==============================================================================

! Convert to Cartesian coordinates
COE0_rad = [COE0(1:2),COE0(3:6)*real(d2r,dk)]
call COE2CART(COE0_rad,R0,V0,GE)

! Output to user
GMST0 = GMST_UNIFORM(MJD0)
aGEO  = (GE*(secsPerSidDay/twopi)**2)**(1._dk/3._dk)
period = twopi*sqrt(COE0(1)**3/GE)/secsPerSidDay

write(*,'(a,g15.8)') 'Tolerance: ',tol
write(*,'(a,i2)') 'Equations: ',eqs

call DPROP_REGULAR(R0,V0,tspan,tstep,cart,int_steps,tot_calls,exitcode)

write(*,'(a,i3)') 'Propagation terminated with exit code = ',exitcode

! ==============================================================================
! 04. PROCESSING AND OUTPUT
! ==============================================================================

! Initialize orbital elements array
npts = size(cart,1)
if (allocated(orb)) deallocate(orb)
allocate(orb(1:npts,1:7))
orb = 0._dk

! End timing BEFORE converting back to orbital elements
call SYSTEM_CLOCK(toc)
cputime = real((toc-tic),dk)/real(rate,dk)
write(*,'(a,g11.4,a)') 'CPU time: ',cputime,' s'
write(id_log,'(a,g11.4,a)') 'CPU time: ',cputime,' s'

! Convert to orbital elements.
! orb(1): MJD,  orb(2): a,  orb(3): e, orb(4): i
! orb(5): Om,   orb(6): om, orb(7): M
do ipt=1,npts
    orb(ipt,1) = cart(ipt,1)    ! Copy MJD
    R = cart(ipt,2:4)
    V = cart(ipt,5:7)
    call CART2COE(R,V,orb(ipt,2:7),GE)
    orb(ipt,4:7) = orb(ipt,4:7)/d2r

end do

! Dump output and copy input files to the output directory
write(id_log,'(a)') 'Dumping output to disk.'

call CREATE_OUT(id_cart)
call CREATE_OUT(id_orb)
call CREATE_OUT(id_stat)
call DUMP_TRAJ(id_cart,npts,cart)
call DUMP_TRAJ(id_orb,npts,orb)

! Write statistics line: calls, steps, CPU time, final time and orbital elements
write(id_stat,100) tot_calls, int_steps, tol, cputime, orb(npts,:)

lifetime_yrs = (cart(npts,1) - cart(1,1))/365.25

write(id_log,'(a23,g11.4)')  'Integration tolerance: ',tol
write(id_log,'(a23,g11.4)')  'Total function calls: ',tot_calls
write(id_log,'(a23,g11.4)')  'Integration steps: ',int_steps
write(id_log,'(a23,g22.15)') 'Lifetime (years): ',lifetime_yrs

call DATE_AND_TIME(date_end,time_end,zone)
write(id_log,'(a)') 'End logging on '//date_end//'T'//time_end//' UTC'&
//zone//'.'

close(id_log)

100 format((2(i10,1x),9(es22.15,1x)))
end program THALASSA
