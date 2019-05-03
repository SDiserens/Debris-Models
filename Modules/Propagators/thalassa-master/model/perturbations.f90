module PERTURBATIONS
! Description:
!    Contains wrapper subroutines to include perturbations in the equations of
!    motion.
!
! Author:
!    Davide Amato
!    Space Dynamics Group - Technical University of Madrid
!    davideamato@email.arizona.edu
!
! ==============================================================================

use KINDS,       only: dk
use NSGRAV
use SUN_MOON
use DRAG_EXPONENTIAL
use US76_PATRIUS
use SRP
use NSGRAV,  only: Cnm,Snm
use AUXILIARIES, only: DU,TU
implicit none

! Jacchia 77 dynamical atmospheric model
external  ::  ISDAMO
! NRLMSISE-00 atmospheric model
external  ::  GTD7,GTD7D,METERS




contains




subroutine PERT_EJ2K(insgrav,isun,imoon,idrag,iF107,iSRP,r,v,rm,t,P_EJ2K,pot,dPot)
! Description:
!    Computes the perturbing acceleration in the EMEJ2000 frame due to a non-sph
!    gravity field, Sun and Moon, drag and solar radiation pressure.
!    Units are DIMENSIONLESS.
!
! Author:
!    Davide Amato
!    The University of Arizona
!    davideamato@email.arizona.edu
!
! Revisions:
!     180608: add variable solar flux.
!     180610: add geodetic longitude and altitude.
!     181006: add truncated third-body acceleration.
!     181203: conversion from TT to UTC for the atmospheric models.
! 
! ==============================================================================

! MODULES
use PHYS_CONST,  only: GE,GE_nd,GS,GM,RE_nd,ERR_constant,secsPerDay,twopi,RE,&
&RS,CD,A2M_Drag,pSRP_1au,au,CR,A2M_SRP,MJD_J1950,GMST_UNIFORM,Kp,Ap,JD2CAL,&
&r2d,cutoff_height,flatt,delta_JD_MJD
use PHYS_CONST,  only: F107DAILY, UTC2TT
use AUXILIARIES, only: DU,TU,MJD0
use AUXILIARIES, only: T2MJD


! VARIABLES
implicit none
! Arguments IN
real(dk),intent(in)  ::  r(1:3),rm           ! Radius vector and its magnitude
real(dk),intent(in)  ::  v(1:3)              ! Velocity vector
real(dk),intent(in)  ::  t                   ! Time (dimensionless)
integer,intent(in)   ::  insgrav,isun        ! Perturbation flags
integer,intent(in)   ::  imoon,idrag,iSRP    ! More perturbation flags
integer,intent(in)   ::  iF107               ! F107 flag
! Perturbing acceleration
real(dk),intent(out)           ::  P_EJ2K(1:3)
! Perturbing potential and its time derivative
real(dk),optional,intent(out)  ::  pot,dPot
! LOCALS
! Non-spherical gravity
real(dk)  ::  gradU_sph(1:3)
real(dk)  ::  p_nsg(1:3)
! Third bodies
real(dk)  ::  r_sun(1:3),v_sun(1:3),p_sun(1:3)
real(dk)  ::  r_moon(1:3),v_moon(1:3),p_moon(1:3)
! Drag, density (US76)
real(dk)  ::  p_drag(1:3)
real(dk)  ::  drag_term
real(dk)  ::  density,pressure,temperature
! Drag and associated q.ties (J77, NRLMSISE-00)
real(dk)  ::  RA,DEC
real(dk)  ::  RA_sun,DEC_sun,r_sun_m
real(dk)  ::  JD_UTC,JD_TT,MJD_UTC,MJD_TT,RJUD,DAFR,GMST,GMST_deg
real(dk)  ::  tempK(1:2),nDens(1:6),wMol
real(dk)  ::  SEC
real(dk)  ::  GLAT_deg,GLONG_deg,RA_deg,STL_hrs
real(dk)  ::  dens_MSIS00(1:6),temp_MSIS00(1:2)
real(dk)  ::  F107
real(dk)  ::  hGeod_D
integer   ::  GDStat
integer   ::  IYD
! Time and date
integer   ::  year,month
real(dk)  ::  dayOfMonth,dayOfYear
! Velocity wrt atmosphere (km/s)
real(dk)  ::  v_rel(1:3),v_atm(1:3),v_relNorm
! Dimensional quantities for drag computation
real(dk)  ::  r_D(1:3),v_D(1:3),h_D
real(dk)  ::  wE_D
! Solar radiation pressure
real(dk)  ::  p_SRP(1:3)


! SOFA routines
external  ::  iau_GC2GDE
real(dk)  ::  iau_GMST06


! ==============================================================================

P_EJ2K = 0._dk

! ==============================================================================
! 01. Potential perturbations
! ==============================================================================

p_nsg = 0._dk
if (insgrav /= 0) then
  if (present(pot) .and. present(dPot)) then
    pot = 0._dk; dPot = 0._dk
    call PINES_NSG(GE_nd,RE_nd,r,t,p_nsg,pot,dPot)
  
  else
    call PINES_NSG(GE_nd,RE_nd,r,t,p_nsg)

  end if
end if

P_EJ2K = P_EJ2K + p_nsg

! ==============================================================================
! 02. Lunisolar perturbations
! ==============================================================================

p_sun = 0._dk; p_moon = 0._dk
if (isun == 1) then
  ! SUN, full acceleration
  call EPHEM(1,DU,TU,t,r_sun,v_sun)
  p_sun = ACC_THBOD_EJ2K_ND(r,r_sun,GE,GS)

else if (isun > 1) then
  ! SUN, truncated acceleration
  call EPHEM(1,DU,TU,t,r_sun,v_sun)
  p_sun = ACC_THBOD_EJ2K_TRUNC_ND(r,r_sun,GE,GS,isun,GslSun)

end if

if (imoon == 1 ) then
  ! MOON, full acceleration
  call EPHEM(2,DU,TU,t,r_moon,v_moon)
  p_moon = ACC_THBOD_EJ2K_ND(r,r_moon,GE,GM)

else if (imoon > 1) then
  ! MOON, truncated acceleration
  call EPHEM(2,DU,TU,t,r_moon,v_moon)
  p_moon = ACC_THBOD_EJ2K_TRUNC_ND(r,r_moon,GE,GM,imoon,GslMoon)

end if

P_EJ2K = p_sun + p_moon + P_EJ2K

! ==============================================================================
! 03. Atmospheric drag
! ==============================================================================
! NOTE:
! Currently, the following approximations are made in the computation of the
! atmospheric drag:
! - Average F10.7 is equal to daily
! - Geomagnetic activity is constant (its value is specified in 
!   ./data/physical_constants.txt)
! - Geodetic height is assumed = geometric height and geodetic
!   latitude/longitude = geometric, i.e. the ellipticity of the Earth is
!   neglected.

p_drag = 0._dk
! Make quantities dimensional
r_D = r*DU; v_D = v*DU*TU
h_D = sqrt(dot_product(r_D,r_D)) - RE
if (idrag /= 0 .and. h_D <= cutoff_height) then
  select case (idrag)
    case (1)
      ! Piecewise exponential density model (Wertz)
      density = ATMOS_VALLADO(h_D)
    
    case (2)
      ! US76 Atmosphere (Fortran porting of the PATRIUS 3.4.1 code)
      call US76_UPPER(h_D,temperature,pressure,density)
    
    case (3)
      ! Jacchia 77 atmosphere (code by V. Carrara - INPE)
      density = 0._dk
      if (h_D <= 2000._dk) then ! Check on max altitude for J77
        ! Right ascension and declination
        RA  = atan2(r(2),r(1))
        RA  = mod(RA + twopi,twopi)
        DEC = asin(r(3)/rm)
        
        ! Ephemerides of the Sun (if they haven't been computed earlier)
        if (isun == 0) then
          call EPHEM(1,DU,TU,t,r_sun,v_sun)
        
        end if
        r_sun_m = sqrt(dot_product(r_sun,r_sun))
        RA_sun  = atan2(r_sun(2),r_sun(1))
        RA_sun  = mod(RA_sun +twopi,twopi)
        DEC_sun = asin(r_sun(3)/r_sun_m)

        MJD_UTC = T2MJD(t)
        MJD_TT  = UTC2TT(MJD_UTC)
        RJUD    = MJD_UTC - MJD_J1950
        DAFR    = RJUD - int(RJUD)
        GMST    = iau_GMST06 ( delta_JD_MJD, MJD_UTC, delta_JD_MJD, MJD_TT )
        F107    = F107DAILY(iF107,MJD_UTC)
        call ISDAMO([RA,DEC,h_D*1.E3_dk],[RA_sun,DEC_sun],&
        & [F107,F107,Kp],RJUD,DAFR,GMST,tempK,nDens,wMol,density)
      
      end if
    
    case (4)
      ! NRLMSISE-00 atmospheric model
      density = 0._dk
      ! Get date and year
      MJD_UTC = T2MJD(t)
      MJD_TT  = UTC2TT(MJD_UTC)
      JD_UTC  = MJD_UTC + delta_JD_MJD
      call JD2CAL(JD_UTC,year,month,dayOfMonth,dayOfYear)
      
      ! Note: year number is ignored in NRLMSISE-00.
      IYD       = int(dayOfYear)
      SEC       = (dayOfYear - IYD)*secsPerDay
      GMST_deg  = iau_GMST06 ( delta_JD_MJD, MJD_UTC, delta_JD_MJD, MJD_TT )*r2d
      RA_deg    = mod(atan2(r(2),r(1)) + twopi, twopi)*r2d
      GLONG_deg = mod(RA_deg - GMST_deg + 360._dk,360._dk)
      STL_hrs   = SEC/3600._dk + GLONG_deg/15._dk
      
      F107      = F107DAILY(iF107,MJD_UTC)
      
      ! Geodetic altitude and latitude
      call iau_GC2GDE(RE,flatt,r_D,RA_deg,GLAT_deg,hGeod_D,GDStat)
      RA_deg   = RA_deg * r2d
      GLAT_deg = GLAT_deg * r2d 

      ! Compute density
      call METERS(.true.)
      if (h_D <= 500._dk) then
        call GTD7(IYD,SEC,hGeod_D,GLAT_deg,GLONG_deg,STL_hrs,F107,F107,Ap,48,&
        &dens_MSIS00,temp_MSIS00)
      
      else
        ! Do not neglect contribution from anomalous oxygen
        call GTD7D(IYD,SEC,hGeod_D,GLAT_deg,GLONG_deg,STL_hrs,F107,F107,Ap,48,&
        &dens_MSIS00,temp_MSIS00)
      
      end if
      density = dens_MSIS00(6)

    end select

    ! Velocity wrt atmosphere
    wE_D = ERR_constant*twopi/secsPerDay
    v_atm = wE_D*[-r_D(2),r_D(1),0._dk]
    v_rel = v_D - v_atm
    v_relNorm = sqrt(dot_product(v_rel,v_rel))
    
    ! Acceleration (in km/s^2)
    drag_term = -0.5_dk*1.E3_dk*CD*A2M_Drag*density
    p_drag    = drag_term*v_relNorm*v_rel
    
    ! Acceleration (non-dimensionalized)
    p_drag = p_drag/(DU*TU**2)

end if

P_EJ2K = p_drag + P_EJ2K

! ==============================================================================
! 04. Solar radiation pressure
! ==============================================================================

p_SRP = 0._dk
if (iSRP /= 0) then
  ! If the Sun gravitational perturbation is disabled, get its ephemerides anyway
  if (isun == 0) then
    call EPHEM(1,DU,TU,t,r_sun,v_sun)

  end if
  ! Computation is in dimensional units (m/s^2).
  p_SRP = SRP_ACC(iSRP,RS,RE,pSRP_1au,au,CR,A2M_SRP,r*DU,r_sun*DU)
  p_SRP = p_SRP/((DU*1.E3_dk)*TU**2)

end if
P_EJ2K = p_SRP + P_EJ2K

end subroutine PERT_EJ2K




end module PERTURBATIONS
