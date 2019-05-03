module DRAG_EXPONENTIAL
    ! Description:
    !    Contains the subroutine to compute the perturbation acceleration due to drag. The
    !    atmospheric model is exponential.

    use KINDS, only: dk
    implicit none
    
    
    
    
    contains
    
    
    
    
    function DRAG_ACC(y_D,wE,RE,CD,A2M)
    ! Description:
    !    Computes the dimensional drag acceleration from the state vector and parameters. Uses
    !    data from Vallado's Fundamentals of Astrodynamics, 4th Edition; data about atmospheric
    !    density is from the US Standard Atmosphere (1976) and CIRA-72 models.
    !
    ! INPUTS:
    !    y_D = dimensional state vector [km,km/s]
    !    wE  = Earth's rotational velocity [rad/s]
    !    RE  = Earth's equatorial radius [km]
    !    CD  = drag coefficient
    !    A2M = area-to-mass ratio [m^2/kg]
    !
    ! OUTPUTS:
    !    DRAG_ACC = dimensional drag acceleration [km/s^2]
    !
    ! VERSIONS:
    !    10/06/2015: v1.
    !    11/06/2015: v2. Add 0-350 km altitude range.
    !    06/04/2017: v3. Add kinds.
    !    29/09/2017: DEPRECATED.
    !
    ! AUTHOR:
    ! Davide Amato
    ! Space Dynamics Group
    ! E.T.S.I.A.E. - UPM
    ! davideamato@email.arizona.edu
    !
    ! =========================================================================================

    ! VARIABLES
    implicit none
    ! Arguments IN
    real(dk),intent(in)  ::  y_D(1:6),wE,RE,CD,A2M
    ! Function definition
    real(dk)         ::  DRAG_ACC(1:3)

    ! Density data
    real(dk)         ::  rho0,h0,scale_H
    real(dk)         ::  rNorm,h,rho
    ! Relative velocity
    real(dk)         ::  v_atm(1:3),v_rel(1:3),v_relNorm
    real(dk)         ::  D_factor

    ! =========================================================================================

    ! =======================================
    ! 1. DENSITY FROM TABULATED DATA
    ! =======================================

    rNorm = sqrt(dot_product(y_D(1:3),y_D(1:3)))
    h = rNorm - RE

    ! Select base and scale heights and nominal density. Density is in kg/m^3.
    if (h >= 0._dk .and. h < 25._dk) then
        h0      = 0._dk
        scale_H = 7.249_dk
        rho0    = 1.225_dk
    else if (h >= 25._dk .and. h < 30._dk) then               ! 0 KM (SEA LEVEL)
        h0      = 25._dk
        scale_H = 6.349_dk
        rho0    = 3.899E-2_dk
    else if (h >= 30._dk .and. h < 40._dk) then
        h0      = 30._dk
        scale_H = 6.682_dk
        rho0    = 1.774E-2_dk
    else if (h >= 40._dk .and. h < 50._dk) then
        h0      = 40._dk
        scale_H = 7.554_dk
        rho0    = 3.972E-3_dk
    else if (h >= 50._dk .and. h < 60._dk) then
        h0      = 50._dk
        scale_H = 8.382_dk
        rho0    = 1.057E-3_dk
    else if (h >= 60._dk .and. h < 70._dk) then
        h0      = 60._dk
        scale_H = 7.714_dk
        rho0    = 3.206E-4_dk
    else if (h >= 70._dk .and. h < 80._dk) then
        h0      = 70._dk
        scale_H = 6.549_dk
        rho0    = 8.770E-5_dk
    else if (h >= 80._dk .and. h < 90._dk) then
        h0      = 80._dk
        scale_H = 5.799_dk
        rho0    = 1.905E-5_dk
    else if (h >= 90._dk .and. h < 100._dk) then
        h0      = 90._dk
        scale_H = 5.382_dk
        rho0    = 3.396E-6_dk
    else if (h >= 100._dk .and. h < 110._dk) then             ! 100 KM
        h0      = 100._dk
        scale_H = 5.877_dk
        rho0    = 5.297E-7_dk
    else if (h >= 110._dk .and. h < 120._dk) then
        h0      = 110._dk
        scale_H = 7.263_dk
        rho0    = 9.661E-8_dk
    else if (h >= 120._dk .and. h < 130._dk) then
        h0      = 120._dk
        scale_H = 9.473_dk
        rho0    = 2.438E-8_dk
    else if (h >= 130._dk .and. h < 140._dk) then
        h0      = 130._dk
        scale_H = 12.636_dk
        rho0    = 8.484E-9_dk
    else if (h >= 140._dk .and. h < 150._dk) then
        h0      = 140._dk
        scale_H = 16.149_dk
        rho0    = 3.845E-9_dk
    else if (h >= 150._dk .and. h < 180._dk) then
        h0      = 150._dk
        scale_H = 22.523_dk
        rho0    = 2.070E-9_dk
    else if (h >= 180._dk .and. h < 200._dk) then
        h0      = 180._dk
        scale_H = 29.740_dk
        rho0    = 5.464E-10_dk
    else if (h >= 200._dk .and. h < 250._dk) then
        h0      = 200._dk
        scale_H = 37.105_dk
        rho0    = 2.789E-10_dk
    else if (h >= 250._dk .and. h < 300._dk) then
        h0      = 250._dk
        scale_H = 45.546_dk
        rho0    = 7.248E-11_dk
    else if (h >= 300._dk .and. h < 350._dk) then             ! 300 KM
        h0      = 300._dk
        scale_H = 53.628_dk
        rho0    = 2.418E-11_dk
    else if (h >= 350._dk .and. h < 400._dk) then
        h0      = 350._dk
        scale_H = 53.298_dk
        rho0    = 9.518E-12_dk
    else if (h >= 400._dk .and. h < 450._dk) then
        h0      = 400._dk
        scale_H = 58.515_dk
        rho0    = 3.725E-12_dk
    else if (h >= 450._dk .and. h < 500._dk) then
        h0      = 450._dk
        scale_H = 60.828_dk
        rho0    = 1.585E-12_dk
    else if (h >= 500._dk .and. h < 600._dk) then
        h0      = 500._dk
        scale_H = 63.822_dk
        rho0    = 6.967E-13_dk
    else if (h >= 600._dk .and. h < 700._dk) then
        h0      = 600._dk
        scale_H = 71.835_dk
        rho0    = 1.454E-13_dk
    else if (h >= 700._dk .and. h < 800._dk) then
        h0      = 700._dk
        scale_H = 88.667_dk
        rho0    = 3.614E-14_dk
    else if (h >= 800._dk .and. h < 900._dk) then
        h0      = 800._dk
        scale_H = 124.64_dk
        rho0    = 1.170E-14_dk
    else if (h >= 900._dk .and. h < 1000._dk) then
        h0      = 900._dk
        scale_H = 181.05_dk
        rho0    = 5.245E-15_dk
    else if (h > 1000.) then                            ! > 1000 KM
        h0      = 1000._dk
        scale_H = 268._dk
        rho0    = 3.019E-15_dk
    end if

    rho = rho0*exp(-(h-h0)/scale_H)

    ! ===============================================
    ! 2. RELATIVE VELOCITY WITH RESPECT TO ATMOSPHERE
    ! ===============================================

    v_atm = wE*[-y_D(2),y_D(1),0._dk]
    v_rel = y_D(4:6) - v_atm
    v_relNorm = norm2(v_rel)

    ! ===============================================
    ! 3. DRAG ACCELERATION
    ! ===============================================

    ! Acceleration is in km/s^2
    D_factor = -0.5_dk*1000._dk*CD*A2M*rho
    DRAG_ACC = D_factor*v_relNorm*v_rel

    end function DRAG_ACC
    
    
    
    
    function ATMOS_VALLADO(h)
    ! Description:
    !    Computes the atmospheric density according to the piecewise exponential
    !    model in: 
    !    D. Vallado, W. D. McClain, "Fundamentals of Astrodynamics", 4th Edition
    !    
    !    Data is originally from the US 1976 Standard Atmosphere and the CIRA-72
    !     model.
    !
    ! ==========================================================================
    
    ! VARIABLES
    implicit none
    ! Arguments IN
    real(dk),intent(in)  ::  h             ! Altitude [km]
    ! Function definition
    real(dk)         ::  ATMOS_VALLADO     ! Density [kg/m^3]
     
    ! Density data
    real(dk)         ::  rho0,h0,scale_H
    
    ! ==========================================================================
    
    ! Select base and scale heights and nominal density. Density is in kg/m^3.
    if (h >= 0._dk .and. h < 25._dk) then
        h0      = 0._dk
        scale_H = 7.249_dk
        rho0    = 1.225_dk
    else if (h >= 25._dk .and. h < 30._dk) then               ! 0 KM (SEA LEVEL)
        h0      = 25._dk
        scale_H = 6.349_dk
        rho0    = 3.899E-2_dk
    else if (h >= 30._dk .and. h < 40._dk) then
        h0      = 30._dk
        scale_H = 6.682_dk
        rho0    = 1.774E-2_dk
    else if (h >= 40._dk .and. h < 50._dk) then
        h0      = 40._dk
        scale_H = 7.554_dk
        rho0    = 3.972E-3_dk
    else if (h >= 50._dk .and. h < 60._dk) then
        h0      = 50._dk
        scale_H = 8.382_dk
        rho0    = 1.057E-3_dk
    else if (h >= 60._dk .and. h < 70._dk) then
        h0      = 60._dk
        scale_H = 7.714_dk
        rho0    = 3.206E-4_dk
    else if (h >= 70._dk .and. h < 80._dk) then
        h0      = 70._dk
        scale_H = 6.549_dk
        rho0    = 8.770E-5_dk
    else if (h >= 80._dk .and. h < 90._dk) then
        h0      = 80._dk
        scale_H = 5.799_dk
        rho0    = 1.905E-5_dk
    else if (h >= 90._dk .and. h < 100._dk) then
        h0      = 90._dk
        scale_H = 5.382_dk
        rho0    = 3.396E-6_dk
    else if (h >= 100._dk .and. h < 110._dk) then             ! 100 KM
        h0      = 100._dk
        scale_H = 5.877_dk
        rho0    = 5.297E-7_dk
    else if (h >= 110._dk .and. h < 120._dk) then
        h0      = 110._dk
        scale_H = 7.263_dk
        rho0    = 9.661E-8_dk
    else if (h >= 120._dk .and. h < 130._dk) then
        h0      = 120._dk
        scale_H = 9.473_dk
        rho0    = 2.438E-8_dk
    else if (h >= 130._dk .and. h < 140._dk) then
        h0      = 130._dk
        scale_H = 12.636_dk
        rho0    = 8.484E-9_dk
    else if (h >= 140._dk .and. h < 150._dk) then
        h0      = 140._dk
        scale_H = 16.149_dk
        rho0    = 3.845E-9_dk
    else if (h >= 150._dk .and. h < 180._dk) then
        h0      = 150._dk
        scale_H = 22.523_dk
        rho0    = 2.070E-9_dk
    else if (h >= 180._dk .and. h < 200._dk) then
        h0      = 180._dk
        scale_H = 29.740_dk
        rho0    = 5.464E-10_dk
    else if (h >= 200._dk .and. h < 250._dk) then
        h0      = 200._dk
        scale_H = 37.105_dk
        rho0    = 2.789E-10_dk
    else if (h >= 250._dk .and. h < 300._dk) then
        h0      = 250._dk
        scale_H = 45.546_dk
        rho0    = 7.248E-11_dk
    else if (h >= 300._dk .and. h < 350._dk) then             ! 300 KM
        h0      = 300._dk
        scale_H = 53.628_dk
        rho0    = 2.418E-11_dk
    else if (h >= 350._dk .and. h < 400._dk) then
        h0      = 350._dk
        scale_H = 53.298_dk
        rho0    = 9.518E-12_dk
    else if (h >= 400._dk .and. h < 450._dk) then
        h0      = 400._dk
        scale_H = 58.515_dk
        rho0    = 3.725E-12_dk
    else if (h >= 450._dk .and. h < 500._dk) then
        h0      = 450._dk
        scale_H = 60.828_dk
        rho0    = 1.585E-12_dk
    else if (h >= 500._dk .and. h < 600._dk) then
        h0      = 500._dk
        scale_H = 63.822_dk
        rho0    = 6.967E-13_dk
    else if (h >= 600._dk .and. h < 700._dk) then
        h0      = 600._dk
        scale_H = 71.835_dk
        rho0    = 1.454E-13_dk
    else if (h >= 700._dk .and. h < 800._dk) then
        h0      = 700._dk
        scale_H = 88.667_dk
        rho0    = 3.614E-14_dk
    else if (h >= 800._dk .and. h < 900._dk) then
        h0      = 800._dk
        scale_H = 124.64_dk
        rho0    = 1.170E-14_dk
    else if (h >= 900._dk .and. h < 1000._dk) then
        h0      = 900._dk
        scale_H = 181.05_dk
        rho0    = 5.245E-15_dk
    else if (h > 1000.) then                            ! > 1000 KM
        h0      = 1000._dk
        scale_H = 268._dk
        rho0    = 3.019E-15_dk
    end if

    ATMOS_VALLADO = rho0*exp(-(h-h0)/scale_H)

    end function ATMOS_VALLADO


end module DRAG_EXPONENTIAL
