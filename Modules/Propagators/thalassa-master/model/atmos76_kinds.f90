!+
MODULE Atmosphere1976
! ---------------------------------------------------------------------------
! PURPOSE - Compute properties of the U.S. Standard Atmosphere 1976
! AUTHORS - Steven S. Pietrobon.
!           Ralph L. Carmichael, Public Domain Aeronautical Software
!
!     REVISION HISTORY
!   DATE  VERS PERSON  STATEMENT OF CHANGES
! 28Feb95  0.1   RLC   Assembled several old codes
!  1Aug00  0.2   RLC   Copied from old Tables76
! 23Aug00  0.3   RLC   Added NitrogenNumber using QUANC8
! 24Aug00  0.4   RLC   Added KineticTemperatureGradient
! 30Aug00  0.5   RLC   Corrected numerous errors
! 30Dec00  0.6   RLC   Adapted UpperAtmosphere from Pietrobon's Unofficial
!                        Australian Standard Atmosphere
! 20Jun17  0.61  D.A.  Corrected bug in the values of density for the upper
!                      atmosphere.
!----------------------------------------------------------------------------

use KINDS, only: dk

IMPLICIT NONE
  CHARACTER(LEN=*),PUBLIC,PARAMETER:: ATM76_VERSION = "0.61 (20 June 2017)"
  REAL(dk),PRIVATE,PARAMETER:: PI = 3.14159265_dk
  REAL(dk),PARAMETER:: REARTH = 6356.766_dk               ! radius of the Earth (km)
  REAL(dk),PARAMETER:: GMR = 34.163195_dk                             ! gas constant
  REAL(dk),PARAMETER:: GZERO = 9.80665_dk !  accel. of gravity, m/sec^2

!  REAL,PARAMETER:: FT2METERS = 0.3048       ! mult. ft. to get meters (exact)
!  REAL,PARAMETER:: KELVIN2RANKINE = 1.8             ! mult K to get deg R
!  REAL,PARAMETER:: PSF2NSM = 47.880258          ! mult lb/sq.ft to get N/sq.m
!  REAL,PARAMETER:: SCF2KCM = 515.379         ! mult slug/cu.ft to get kg/cu.m
  REAL(dk),PARAMETER:: TZERO = 288.15_dk                ! temperature at sealevel, K
  REAL(dk),PARAMETER:: PZERO = 101325.0_dk            ! pressure at sealevel, N/sq.m
  REAL(dk),PARAMETER:: RHOZERO = 1.2250_dk            ! density at sealevel, kg/cu.m
  REAL(dk),PARAMETER:: RSTAR = 8314.32_dk       ! perfect gas constant, N-m/(kmol-K)
  REAL(dk),PARAMETER:: ASOUNDZERO = 340.294_dk   ! speed of sound at sealevel, m/sec

  REAL(dk),PARAMETER:: BETAVISC = 1.458E-6_dk    ! viscosity term, N s/(sq.m sqrt(K)
  REAL(dk),PARAMETER:: SUTH = 110.4_dk                    ! Sutherland's constant, K


  REAL(dk),PARAMETER:: MZERO      = 28.9644_dk ! molecular weight of air at sealevel
!  REAL,PARAMETER:: AVOGADRO =  6.022169E26        ! 1/kmol, Avogadro constant
!  REAL,PARAMETER:: BOLTZMANN = 1.380622E-23        ! Nm/K, Boltzmann constant


! TABLE 5 - DEFINITION OF KINETIC TEMPERATURE FROM 86 km to 1000 km
  REAL(dk),PARAMETER:: Z7 =  86.0_dk,  T7=186.8673_dk
  REAL(dk),PARAMETER:: z8 =  91.0_dk,  T8=T7
  REAL(dk),PARAMETER:: Z9 = 110.0_dk,  T9=240.0_dk
  REAL(dk),PARAMETER:: Z10= 120.0_dk, T10=360.0_dk
!  REAL,PARAMETER:: Z11= 500.0, T11=999.2356   ! not used
  REAL(dk),PARAMETER:: Z12=1000.0_dk, T12=1000.0_dk


  REAL(dk),PARAMETER:: FT2METERS = 0.3048_dk               ! mult. ft. to get meters (exact)
  REAL(dk),PARAMETER:: KELVIN2RANKINE = 1.8_dk             ! mult deg K to get deg R
  REAL(dk),PARAMETER:: PSF2NSM = 47.880258_dk          ! mult lb/sq.ft to get N/sq.m
  REAL(dk),PARAMETER:: SCF2KCM = 515.379_dk         ! mult slug/cu.ft to get kg/cu.m




CONTAINS

!+
FUNCTION EvaluateCubic(a,fa,fpa, b,fb,fpb, u) RESULT(fu)
! ---------------------------------------------------------------------------
! PURPOSE - Evaluate a cubic polynomial defined by the function and the
!   1st derivative at two points
  REAL(dk),INTENT(IN):: u   ! point where function is to be evaluated
  REAL(dk),INTENT(IN):: a,fa,fpa   ! a, f(a), f'(a)  at first point
  REAL(dk),INTENT(IN):: b,fb,fpb   ! b, f(b), f'(b)  at second point
  REAL(dk):: fu                    ! computed value of f(u)

  REAL(dk):: d,t,p
!----------------------------------------------------------------------------
  d=(fb-fa)/(b-a)
  t=(u-a)/(b-a)
  p=1.0_dk-t

  fu = p*fa + t*fb - p*t*(b-a)*(p*(d-fpa)-t*(d-fpb))
  RETURN
END Function EvaluateCubic   ! ----------------------------------------------

!+
FUNCTION KineticTemperature(z) RESULT(t)
!   -------------------------------------------------------------------------
! PURPOSE - Compute kinetic temperature above 86 km.

  REAL(dk),INTENT(IN)::  z     ! geometric altitude, km.                        
  REAL(dk):: t     ! kinetic temperature, K

  REAL(dk),PARAMETER:: C1 = -76.3232_dk  ! uppercase A in document
  REAL(dk),PARAMETER:: C2 = 19.9429_dk   ! lowercase a in document
  REAL(dk),PARAMETER:: C3 = 12.0_dk
  REAL(dk),PARAMETER:: C4 = 0.01875_dk   ! lambda in document
  REAL(dk),PARAMETER:: TC = 263.1905_dk

  REAL(dk):: xx,yy
!----------------------------------------------------------------------------
  IF (z <= Z8) THEN
    t=T7
  ELSE IF (z < Z9) THEN  
    xx=(z-Z8)/C2                        ! from Appendix B, p.223
    yy=SQRT(1.0_dk-xx*xx)
    t=TC+C1*yy
  ELSE IF (z <= Z10) THEN
    t=T9+C3*(z-Z9)
  ELSE
    xx=(REARTH+Z10)/(REARTH+z)
    yy=(T12-T10)*EXP(-C4*(z-Z10)*xx)
    t=T12-yy
  END IF

  RETURN
END Function KineticTemperature   ! -----------------------------------------


!+
SUBROUTINE UpperAtmosphere(alt, sigma, delta, theta)
!   -------------------------------------------------------------------------
! PURPOSE - Compute the properties of the 1976 standard atmosphere from
!   86 km. to 1000 km.

  IMPLICIT NONE
!============================================================================
!     A R G U M E N T S                                                     |
!============================================================================
  REAL(dk),INTENT(IN)::  alt    ! geometric altitude, km.                        
  REAL(dk),INTENT(OUT):: sigma  ! density/sea-level standard density              
  REAL(dk),INTENT(OUT):: delta  ! pressure/sea-level standard pressure           
  REAL(dk),INTENT(OUT):: theta  ! temperature/sea-level standard temperature
!============================================================================
!     L O C A L   C O N S T A N T S                                         |
!============================================================================

! altitude table (m)
  REAL(dk),PARAMETER,DIMENSION(23):: Z = (/      &
     86._dk,  93._dk, 100._dk, 107._dk, 114._dk, &
    121._dk, 128._dk, 135._dk, 142._dk, 150._dk, &
    160._dk, 170._dk, 180._dk, 190._dk, 200._dk, &
    250._dk, 300._dk, 400._dk, &
    500._dk, 600._dk, 700._dk, 800._dk, 1000._dk /)

! pressure table  (Pa)
!  REAL,PARAMETER,DIMENSION(SIZE(Z)):: P = (/                                &
!    3.7338e-1, 1.0801e-1, 3.2011e-2, 1.0751E-2, 4.4473e-3, &
!    2.3402e-3, 1.4183e-3, 9.3572e-4, 6.5297e-4, 4.5422e-4, &
!    3.0397e-4, 2.1212e-4, 1.5273e-4, 1.1267e-4, 8.4743e-5, &
!    2.4767e-5, 8.7704e-6, 1.4518e-6, &
!    3.0236e-7, 8.2130e-8, 3.1908e-8, 1.7036e-8, 7.5138e-9 /)

! density table  kg/m**3
!  REAL,PARAMETER,DIMENSION(SIZE(Z)):: RHO = (/                              &
!    6.9579e-06, 1.9997e-06, 5.6041e-07, 1.6426E-07, 4.9752e-08, &
!    1.9768e-08, 9.7173e-09, 5.4652e-09, 3.3580e-09, 2.0757e-09, &
!    1.2332e-09, 7.8155e-10, 5.1944e-10, 3.5808e-10, 2.5409e-10, &
!    6.0732e-11, 1.9160e-11, 2.8031e-12, &
!    5.2159e-13, 1.1369e-13, 3.0698e-14, 1.1361e-14, 3.5614e-15 /)


!fit2000
  REAL(dk),PARAMETER,DIMENSION(SIZE(Z)):: LOGP = (/               &
  -0.985159_dk,  -2.225531_dk,  -3.441676_dk,  -4.532756_dk,  -5.415458_dk,  &
  -6.057519_dk,  -6.558296_dk,  -6.974194_dk,  -7.333980_dk,  -7.696929_dk,  &
  -8.098581_dk,  -8.458359_dk,  -8.786839_dk,  -9.091047_dk,  -9.375888_dk,  &
 -10.605998_dk, -11.644128_dk, -13.442706_dk, -15.011647_dk, -16.314962_dk,  &
 -17.260408_dk, -17.887938_dk, -18.706524_dk /)

 REAL(dk),PARAMETER,DIMENSION(SIZE(Z)):: LOGRHO = (/             &
 -11.875633_dk, -13.122514_dk, -14.394597_dk, -15.621816_dk, -16.816216_dk,  &
 -17.739201_dk, -18.449358_dk, -19.024864_dk, -19.511921_dk, -19.992968_dk,  &
 -20.513653_dk, -20.969742_dk, -21.378269_dk, -21.750265_dk, -22.093332_dk,  &
 -23.524549_dk, -24.678196_dk, -26.600296_dk, -28.281895_dk, -29.805302_dk,  &
 -31.114578_dk, -32.108589_dk, -33.268623_dk /)

  REAL(dk),PARAMETER,DIMENSION(SIZE(Z)):: DLOGPDZ = (/            &
  -0.177700_dk,  -0.176950_dk,  -0.167294_dk,  -0.142686_dk,  -0.107868_dk,  &
  -0.079313_dk,  -0.064668_dk,  -0.054876_dk,  -0.048264_dk,  -0.042767_dk,  &
  -0.037847_dk,  -0.034273_dk,  -0.031539_dk,  -0.029378_dk,  -0.027663_dk,  &
  -0.022218_dk,  -0.019561_dk,  -0.016734_dk,  -0.014530_dk,  -0.011315_dk,  &
  -0.007673_dk,  -0.005181_dk,  -0.003500_dk /)

  REAL(dk),PARAMETER,DIMENSION(SIZE(Z)):: DLOGRHODZ = (/          & 
  -0.177900_dk,  -0.180782_dk,  -0.178528_dk,  -0.176236_dk,  -0.154366_dk,  &
  -0.113750_dk,  -0.090551_dk,  -0.075044_dk,  -0.064657_dk,  -0.056087_dk,  &
  -0.048485_dk,  -0.043005_dk,  -0.038879_dk,  -0.035637_dk,  -0.033094_dk,  &
  -0.025162_dk,  -0.021349_dk,  -0.017682_dk,  -0.016035_dk,  -0.014330_dk,  &
  -0.011626_dk,  -0.008265_dk,  -0.004200_dk /)


!============================================================================
!     L O C A L   V A R I A B L E S                                         |
!============================================================================
  INTEGER:: i,j,k                                                  ! counters
!  REAL:: h                                       ! geopotential altitude (km)
!  REAL:: tgrad, tbase      ! temperature gradient and base temp of this layer
!  REAL:: tlocal                                           ! local temperature
!  REAL:: deltah                             ! height above base of this layer

  REAL(dk):: p,rho
!----------------------------------------------------------------------------

  IF (alt >= Z(SIZE(Z))) THEN          ! trap altitudes greater than 1000 km.
    ! delta=1.E-20_dk
    ! sigma=1.E-21_dk
    delta=0._dk
    sigma=0._dk
    theta=1000.0_dk/TZERO
    RETURN
  END IF

  i=1 
  j=SIZE(Z)                                    ! setting up for binary search
  DO
    k=(i+j)/2                                              ! integer division
    IF (alt < Z(k)) THEN
      j=k
    ELSE
      i=k
    END IF   
    IF (j <= i+1) EXIT
  END DO
  
  p=EXP(EvaluateCubic(Z(i),LOGP(i),DLOGPDZ(i), &
                      Z(i+1),LOGP(i+1),DLOGPDZ(i+1), alt))
  delta=p/PZERO
  
  rho=EXP(EvaluateCubic(Z(i),LOGRHO(i),DLOGRHODZ(i), &
                      Z(i+1),LOGRHO(i+1),DLOGRHODZ(i+1), alt))
  sigma=rho/RHOZERO

  theta=KineticTemperature(alt)/TZERO
  RETURN
END Subroutine UpperAtmosphere   ! ------------------------------------------

!+
SUBROUTINE LowerAtmosphere(alt, sigma, delta, theta)
!   -------------------------------------------------------------------------
! PURPOSE - Compute the properties of the 1976 standard atmosphere to 86 km.

  IMPLICIT NONE
!============================================================================
!     A R G U M E N T S                                                     |
!============================================================================
  REAL(dk),INTENT(IN)::  alt    ! geometric altitude, km.                        
  REAL(dk),INTENT(OUT):: sigma  ! density/sea-level standard density              
  REAL(dk),INTENT(OUT):: delta  ! pressure/sea-level standard pressure           
  REAL(dk),INTENT(OUT):: theta  ! temperature/sea-level standard temperature
!============================================================================
!     L O C A L   C O N S T A N T S                                         |
!============================================================================
  REAL(dk),PARAMETER:: REARTH = 6369.0_dk                 ! radius of the Earth (km)
  REAL(dk),PARAMETER:: GMR = 34.163195_dk                             ! gas constant
  INTEGER,PARAMETER:: NTAB=8       ! number of entries in the defining tables
!============================================================================
!     L O C A L   V A R I A B L E S                                         |
!============================================================================
  INTEGER:: i,j,k                                                  ! counters
  REAL(dk):: h                                       ! geopotential altitude (km)
  REAL(dk):: tgrad, tbase      ! temperature gradient and base temp of this layer
  REAL(dk):: tlocal                                           ! local temperature
  REAL(dk):: deltah                             ! height above base of this layer
!============================================================================
!     L O C A L   A R R A Y S   ( 1 9 7 6   S T D.  A T M O S P H E R E )   |
!============================================================================
  REAL(dk),DIMENSION(NTAB),PARAMETER:: htab= &
                          (/0.0_dk, 11.0_dk, 20.0_dk, 32.0_dk, 47.0_dk, 51.0_dk&
                          &, 71.0_dk, 84.852_dk/)
  REAL(dk),DIMENSION(NTAB),PARAMETER:: ttab= &
          (/288.15_dk, 216.65_dk, 216.65_dk, 228.65_dk, 270.65_dk, 270.65_dk,&
          & 214.65_dk, 186.946_dk/)
  REAL(dk),DIMENSION(NTAB),PARAMETER:: ptab= &
               (/1.0_dk, 2.233611E-1_dk, 5.403295E-2_dk, 8.5666784E-3_dk, &
               1.0945601E-3_dk, 6.6063531E-4_dk, 3.9046834E-5_dk, 3.68501E-6_dk/)
  REAL(dk),DIMENSION(NTAB),PARAMETER:: gtab= &
                                (/-6.5_dk, 0.0_dk, 1.0_dk, 2.8_dk, 0.0_dk, &
                                &-2.8_dk, -2.0_dk, 0.0_dk/)
!----------------------------------------------------------------------------
  h=alt*REARTH/(alt+REARTH)      ! convert geometric to geopotential altitude

  i=1 
  j=NTAB                                  ! setting up for binary search
  DO
    k=(i+j)/2                                              ! integer division
    IF (h < htab(k)) THEN
      j=k
    ELSE
      i=k
    END IF   
    IF (j <= i+1) EXIT
  END DO

  tgrad=gtab(i)                                     ! i will be in 1...NTAB-1
  tbase=ttab(i)
  deltah=h-htab(i)
  tlocal=tbase+tgrad*deltah
  theta=tlocal/ttab(1)                                    ! temperature ratio

  IF (tgrad == 0.0_dk) THEN                                     ! pressure ratio
    delta=ptab(i)*EXP(-GMR*deltah/tbase)
  ELSE
    delta=ptab(i)*(tbase/tlocal)**(GMR/tgrad)
  END IF

  sigma=delta/theta                                           ! density ratio
  RETURN
END Subroutine LowerAtmosphere   ! ------------------------------------------

!+
SUBROUTINE SimpleAtmosphere(alt,sigma,delta,theta)
!   -------------------------------------------------------------------------
! PURPOSE - Compute the characteristics of the atmosphere below 20 km.

! NOTES-Correct to 20 km. Only approximate above there

  IMPLICIT NONE
!============================================================================
!     A R G U M E N T S                                                     |
!============================================================================
  REAL(dk),INTENT(IN)::  alt    ! geometric altitude, km.
  REAL(dk),INTENT(OUT):: sigma  ! density/sea-level standard density             
  REAL(dk),INTENT(OUT):: delta  ! pressure/sea-level standard pressure            
  REAL(dk),INTENT(OUT):: theta  ! temperature/sea-level standard temperature   
!============================================================================
!     L O C A L   C O N S T A N T S                                         |
!============================================================================
  REAL(dk),PARAMETER:: REARTH = 6369.0_dk                ! radius of the Earth (km)
  REAL(dk),PARAMETER:: GMR = 34.163195_dk                            ! gas constant
!============================================================================
!     L O C A L   V A R I A B L E S                                         |
!============================================================================
  REAL(dk):: h   ! geopotential altitude
!----------------------------------------------------------------------------
  h=alt*REARTH/(alt+REARTH)      ! convert geometric to geopotential altitude

  IF (h < 11.0_dk) THEN
    theta=1.0_dk+(-6.5_dk/288.15_dk)*h                               ! Troposphere
    delta=theta**(GMR/6.5_dk)
  ELSE
    theta=216.65_dk/288.15_dk                                        ! Stratosphere
    delta=0.2233611_dk*EXP(-GMR*(h-11.0_dk)/216.65_dk)
  END IF

  sigma=delta/theta
  RETURN
END Subroutine SimpleAtmosphere   ! -----------------------------------------

!+
FUNCTION Viscosity(theta) RESULT(visc)
!   -------------------------------------------------------------------------
! PURPOSE - Compute viscosity using Sutherland's formula.
!        Returns viscosity in kg/(meter-sec)

  IMPLICIT NONE
  REAL(dk),INTENT(IN) :: theta                ! temperature/sea-level temperature  
  REAL(dk):: visc
  REAL(dk):: temp                              ! temperature in deg Kelvin
!----------------------------------------------------------------------------
  temp=TZERO*theta
  visc=BETAVISC*Sqrt(temp*temp*temp)/(temp+SUTH)
  RETURN
END Function Viscosity   ! --------------------------------------------

!+
SUBROUTINE Atmosphere(alt,sigma,delta,theta)
!   -------------------------------------------------------------------------
! PURPOSE - Compute the characteristics of the U.S. Standard Atmosphere 1976

  IMPLICIT NONE
  REAL(dk),INTENT(IN)::  alt    ! geometric altitude, km.
  REAL(dk),INTENT(OUT):: sigma  ! density/sea-level standard density             
  REAL(dk),INTENT(OUT):: delta  ! pressure/sea-level standard pressure            
  REAL(dk),INTENT(OUT):: theta  ! temperature/sea-level standard temperature   
!============================================================================
  IF (alt > 86.0_dk) THEN
    CALL UpperAtmosphere(alt,sigma,delta,theta)
  ELSE
    CALL LowerAtmosphere(alt,sigma,delta,theta)
  END IF
  RETURN
END Subroutine Atmosphere   ! -----------------------------------------------

END Module Atmosphere1976   ! ===============================================

