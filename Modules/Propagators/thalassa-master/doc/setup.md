# General requirements
THALASSA has been developed and tested on MacOS and Linux, using the ``gfortran`` compiler. No support for Windows or other compilers (such as ``ifort``) is available, although an experienced user should be able to make the necessary (few) modifications to the makefile and code.

THALASSA requires a ``gfortran`` installation (version 6 or higher), the NAIF ``SPICE`` Fortran toolkit (version N0065 or higher) and the associated SPICE kernels (see the next section), and the [``SOFA`` (Standards Of Fundamental Astronomy) Fortran 77 library](http://www.iausofa.org/current_F.html#Downloads "SOFA library download").

# SPICE toolkit overview
SPICE is a powerful library containing a wide array of subroutines for space mission design, observation planning, and astrodynamics. SPICE is used in THALASSA to read the JPL ephemerides which provide the position of the Sun and the Moon. This is necessary to accurately compute perturbations due to their gravitational field, solar radiation pressure, and atmospheric drag.

In order to run THALASSA, you first need a working SPICE installation. The SPICE developers at the NASA Navigation and Ancillary Information Facility (NAIF) provide [excellent documentation](https://naif.jpl.nasa.gov/naif/documentation.html "NAIF Documentation") to aid in the installation of SPICE, besides [a set of tutorials](https://naif.jpl.nasa.gov/naif/tutorials.html "NAIF tutorials").

The latest version of the toolkit, N0066, is downloadable from [here](https://naif.jpl.nasa.gov/naif/toolkit_FORTRAN.html "SPICE Fortran toolkit").

# Installation
1.  Clone the repository into a directory of your choice,

        git clone https://gitlab.com/souvlaki/thalassa.git/ thalassa_dir

2.  Edit the ``makefile`` by assigning the path to the SPICE and SOFA libraries (``spicelib.a`` and ``libsofa.a``) to the ``LIBS`` variable:

        LIBS = $SPICE_PATH/toolkit/lib/spicelib.a $SOFA_PATH/lib/libsofa.a

3.  Also edit ``data/kernels_to_load.furnsh``, inserting the path to the required SPICE kernels in the ``PATH_VALUES`` variable.
``THALASSA`` requires three types of kernels:
  *  [SPK kernel(s) for the JPL Developmental Ephemerides](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/ "SPK kernels"). By default, THALASSA uses DE431,  
  *  [PCK kernel](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/ "PCK kernels") containing the planetary masses,
  *  [NAIF leapseconds kernel](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/ "LSK kernels").  
  The default names for these kernels are ``de431_part-1.bsp``, ``de431_part-2.bsp``, ``naif0012.tls``, ``gm_de431.tpc``. Note that the SPK kernels containing the Solar System ephemerides can be quite large (~3 GB). Note that the user can choose different SPK, PCK and leapseconds kernels by specifying their path in ``data/kernels_to_load.furnsh``. More information is available in the NAIF SPICE tutorials, in particular [the intro to kernels](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/Tutorials/pdf/individual_docs/12_intro_to_kernels.pdf "NAIF tutorials - Intro to Kernels").

4.  Compile the code by running ``make`` in the THALASSA root folder, which generates the ``thalassa.x`` executable.

You're all set and ready to go! :thumbsup: