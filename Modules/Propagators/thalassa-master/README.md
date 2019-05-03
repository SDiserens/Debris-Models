# Introduction to THALASSA
THALASSA is a Fortran orbit propagation code for bodies in the Earth-Moon-Sun system. It works by integrating either Newtonian equations in Cartesian coordinates or regularized equations of motion with the LSODAR (Livermore Solver for Ordinary Differential equations with Automatic Root-finding).

THALASSA is a command-line tool, and has been developed and tested on MacOS and Ubuntu Linux platforms, using the ``gfortran`` compiler. Attention has been paid to avoid using advanced Fortran constructs: while they greatly improve the capabilities of the language, their portability has been found to be problematic. This might change in the future.

The repository also includes some Python3 scripts to perform batch propagations. This feature is currently experimental, but it shouldn't be too difficult for a Python user to generalize the scripts to perform batch propagations on discrete grids of orbital elements.

Details on the mathematical fundamentals of THALASSA are contained in [Amato et al., 2018](#Amato2018).

# Giving credit
If you use THALASSA in your research, please consider giving credit by citing the article specifying the mathematical fundamentals of THALASSA, [Amato et al., 2018](#Amato2018), and/or acknowledging this repository.

# THALASSA User Guide
THALASSA reads settings and initial conditions from two text files. Their paths can be specified as arguments to the THALASSA executable,

    ./thalassa.x [settings_file] [object_file]

If no arguments are provided, THALASSA will read settings and initial conditions from the files `./in/input.txt` and `./in/object.txt`. Sample files are included in the repository.

Fortran is quite strict when interpreting text. When changing the input files, make sure to respect *all* columns and *do not* insert additional rows without changing the code. Explanatory comments in the files are preceded by pound characters.

## Initial conditions file
The initial conditions file contains the initial orbital elements and physical characteristics of the object to be propagated. The orbital elements are expressed in the EMEJ2000 frame; all the epochs are in Terrestrial Time [(Montenbruck and Gill, 2000)](#Montenbruck2000).

In addition to the initial orbital elements, the user also assigns the object mass, areas used in the calculation of atmospheric drag and SRP accelerations, coefficient of drag, and radiation pressure coefficient.

## Settings file
The settings file is divided in four sections.

### Physical model
The first section contains integer flags that allow the user to tune the parameters of the physical model used for the propagation. The meaning of the flags and their allowed values are:
*  `insgrav`: 1 toggles non-spherical gravity, 0 otherwise
*  `isun`: values >1 are interpreted as the order of the Legendre expansion for the solar gravitational perturbing acceleration. 1 toggles the acceleration using the full expression in rectangular coordinates. 0 disables the perturbation. See [Amato et al. (2018)](#Amato2018) for details.
*  `imoon`: values >1 are interpreted as the order of the Legendre expansion for the lunar gravitational perturbing acceleration. 1 toggles the acceleration using the full expression in rectangular coordinates. 0 disables the perturbation. See [Amato et al. (2018)](#Amato2018) for details.
*  `idrag`: select atmospheric model. 0 = no drag, 1 = patched exponential model [(table 8-4 of Vallado and McClain, 2013)](#Vallado2013), 2 = US 1976 Standard Atmosphere [(NASA et al., 1976)](#US1976), 3 = Jacchia 1977 [(Jacchia, 1977)](#Jacchia1977), 4 = NRLMSISE-00 [(Picone et al., 2000)](#Picone2000).
*  `iF107`: 1 toggles variable F10.7 flux, 0 uses the constant value specified in `data/physical_constants.txt`
*  `iSRP`: select SRP model. 0 = no SRP, 1 = SRP with no eclipses, 2 = SRP with conical shadow using the $\nu$ factor from [Montenbruck and Gill (2000)](#Montenbruck2000).
*  `iephem`: select the source of ephemerides of the Sun and the Moon. 1 uses SPICE-read ephemerides (DE431 by default), 2 uses a set of simplified analytical ephemerides by [Meeus (1998)](#Meeus1998).
*  `gdeg`: selects the degree of the Earth's gravitational field (up to 95, although a maximum degree of 15 is recommended).
*  `gord`: selects the order of the Earth's gravitational field (has to be less than min(`gdeg`,95)).

### Integration
The second section tunes the parameters of the numerical solver, LSODAR [(Radhakrishnan and Hindmarsh, 1993)](#Radakrishnan1993).
*  `tol`: local truncation error tolerance. Value should be between 1E-4 and 1E-15. This is the main parameter affecting the accuracy of integration.
*  `tspan`: time span of the integration, in days.
*  `tstep`: output time step, in days. Note that the time step *does not* affect the integration accuracy.
*  `mxstep`: maximum number of output steps. Users should not change this value apart from exceptional situations.
*  `imcol`: 1 toggles the check for collision with the Moon, 0 disables it. Activating the check roughly doubles the CPU time needed to compute lunar perturbations, therefore activate the check only if necessary.

### Equations of motion
The third section only contains the `eqs` flag, which selects the set of equations of motion to be integrated. The value of `eqs` corresponds to the following equations:
1.  Cowell formulation (Newtonian equations in Cartesian coordinates)
2.  EDromo orbital elements, including physical time as a dependent variable [(Baù et al., 2015)](#Bau2015)
3.  EDromo orbital elements, including the constant time element as a dependent variable
4.  EDromo orbital elements, including the linear time element as a dependent variable
5.  Kustaanheimo-Stiefel coordinates, including the physical time as a dependent variable [(section 9 of Stiefel and Scheifele, 1971)](#Stiefel1971)
6.  Kustaanheimo-Stiefel coordinates, including the linear time element as a dependent variable
7.  Stiefel-Scheifel orbital elements, including the physical time as a dependent variable [(section 19 of Stiefel and Scheifele, 1971)](#Stiefel1971)
8.  Stiefel-Scheifel orbital elements, including the linear time element as a dependent variable

The choice of equations depends on the type of orbit being integrated. For LEOs and MEOs, sets 2 to 6 are recommended. Sets 2 to 4 are particularly efficient for HEOs but should be avoided if there's any chance for the orbital energy to change sign, as the integration will fail in such a case.
As a rule of thumb, weakly-perturbed orbits can be integrated most efficiently by using orbital elements.
Strongly-perturbed orbits should be integrated using coordinates, i.e. sets 1, 5, 6.

## Physical data files
The directory `data/` stores files containing information on the physical model used by THALASSA. `data/earth_potential/GRIM5-S1.txt` contains the harmonic coefficients of the GRIM5-S1 potential, in its native format.
The file `data/physical_constants.txt` contains several astronomical and physical constants that are used during the integration.
The constant values of the solar flux and of the planetary index and amplitude are specified here, along with the height at which the orbiter is considered to have re-entered the atmosphere of the Earth.

### Output
The last section contains settings for the output of THALASSA.
*  `verb`: 1 toggles the printing of the current propagation progress, 0 otherwise
*  `out`:  Full path to the directory where THALASSA saves the output files. The path **always** starts at column 5, and ends with a `/`.

It is recommended to untoggle the `verb` flag if THALASSA is used to propagate batches of initial conditions. Failure to do so could unnecessarily clutter `stdout`.

## THALASSA output
THALASSA is launched by executing `thalassa.x` as specified above.

![Launching THALASSA](/uploads/f2f23ecd72642545bd1774f31ca36602/thalassa_instructions.gif)

The code will write the files `cart.dat` and `orbels.dat` to the directory specified by the user. These contain the numerically integrated trajectory in cartesian coordinates and orbital elements respectively, in the EMEJ2000 reference frame.
Additionally, the code writes a file `stats.dat` containing integration statistics along with the value of the orbital elements at the final epoch, and a `propagation.log` file which contains diagnostics for the current propagation.

You should check the `stats.dat` file for any errors that might have taken place during the execution. In particular, THALASSA will write to the log file and to stdout the exit code of the propagation. This is an integer number with the following meaning:
* `0`: nominal exit, reached end time specified in `input.txt`
* `1`: nominal exit, collided with the Earth (atmospheric re-entry).
* `-2`: floating point exception, detected NaNs in the state vector. This is usually caused by the orbit having become hyperbolic when using EDromo, or in some more exotic cases due to a solver tolerance that's too large.
* `-3`: maximum number of steps reached. Try specifying a larger time step in the input file.
* `-10`: unknown exception, try debugging to check what's the problem.

## References
1.  <a name="Amato2018"></a>Amato, D., Bombardelli, C., Baù, G., Morand, V., and Rosengren, A. J. "Non-averaged regularized formulations as an alternative to semi-analytical orbit propagation methods". Submitted to Celestial Mechanics and Dynamical Astronomy, 2018.
2.  <a name="Montenbruck2000"></a>Montenbruck, O., and Gill, E. "Satellite Orbits. Models, Methods, and Applications". Springer-Verlag Berlin Heidelberg, 2000.
3.  <a name="Vallado2013"></a>Vallado, D. A., and McClain, W. D. "Fundamentals of Astrodynamics and Applications". Microcosm Press, 2013.
4. <a name="US1976"></a> NASA, NOAA, and US Air Force, "U.S. Standard Atmosphere, 1976". Technical Report NASA-TM-X-74335, October 1976.
5. <a name="Jacchia1977"></a> Jacchia, L. G. "Thermospheric Temperature, Density, and Composition: New Models". SAO Special Report, **375**, 1977.
6. <a name="Picone2000"></a> Picone, J. M., Hedin, A. E., Drob, D .P., and Aikin, A. C. "NRLMSISE-00 empirical model of the atmosphere: Statistical comparisons and scientific issues". Journal of Geophysical Research: Space Physics, **107**(A12):15–1–16, 2002.
7.  <a name="Meeus1998"></a>Meeus, J. "Astronomical Algorithms", 2nd Ed. Willmann-Bell, 1998.
8.  <a name="Bau2015"></a>Baù, G., Bombardelli, C., Peláez, J., and Lorenzini, E. "Non-singular orbital elements for special perturbations in the two-body problem". Monthly Notices of the Royal Astronomical Society **454**, pp. 2890-2908, 2015.
9.  <a name="Radhakrishnan1993"></a> Radhakrishnan, K. and Hindmarsh, A. C. "Description and use of LSODE, the Livermore Solver for Ordinary Differential Equations". NASA Reference Publication 1327, Lawrence Livermore National Laboratory Report UCRL-ID-113855, 1993.
10. <a name="Stiefel1971"></a> Stiefel E. G. and Scheifele G. "Linear and Regular Celestial Mechanics". Springer-Verlag New York Heidelberg Berlin, 1971.