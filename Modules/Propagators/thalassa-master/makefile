# Compiler
FC = gfortran

# Compile flags
FCFLAGS = -c -O
FLFLAGS = -O -no-pie -o
# Link flags
LIBS = ~/Documents/Codes/SPICE/toolkit/lib/spicelib.a \
~/Documents/Codes/sofa/lib/libsofa.a

# Numerical integrators
SLSODAR = opksa1.o opksa2.o opksmain.o
DLSODAR = opkda1.o opkda2.o opkdmain.o
FORMUL  = cowell.o edromo.o ks.o stische.o regular_aux.o
J77     = atsu.o soflud.o ja77.o

# Source and object files
OBJECTS = kinds.o phys_const.o settings.o io.o kepler.o cart_coe.o nsgrav.o \
sun_moon.o drag_exponential.o US76_PATRIUS.o $(J77) nrlmsise00_sub.o srp.o \
perturbations.o initialize.o integrate.o $(SLSODAR) $(DLSODAR) $(FORMUL) \
propagate.o auxiliaries.o thalassa_main.o

# Binaries
thalassa.x: thalassa_main.o
	$(FC) $(FLFLAGS) thalassa.x $(OBJECTS) $(LIBS)

# Object files
kinds.o: kinds.f90
	$(FC) $(FCFLAGS) kinds.f90

phys_const.o: phys_const.f90 kinds.o settings.o
	$(FC) $(FCFLAGS) phys_const.f90

settings.o: settings.f90 kinds.o
	$(FC) $(FCFLAGS) settings.f90

auxiliaries.o: auxiliaries.f90 kinds.o phys_const.o
	$(FC) $(FCFLAGS) auxiliaries.f90

io.o: io.f90 kinds.o settings.o phys_const.o
	$(FC) $(FCFLAGS) io.f90

kepler.o: kepler.f90 kinds.o
	$(FC) $(FCFLAGS) kepler.f90

cart_coe.o: cart_coe.f90 phys_const.o kepler.o kinds.o
	$(FC) $(FCFLAGS) cart_coe.f90

nsgrav.o: ./model/nsgrav.f90 kinds.o settings.o phys_const.o auxiliaries.o io.o
	$(FC) $(FCFLAGS) ./model/nsgrav.f90

sun_moon.o: ./model/sun_moon.f90 kinds.o phys_const.o settings.o auxiliaries.o
	$(FC) $(FCFLAGS) ./model/sun_moon.f90

drag_exponential.o: ./model/drag_exponential.f90 kinds.o
	$(FC) $(FCFLAGS) ./model/drag_exponential.f90

US76_PATRIUS.o: ./model/US76_PATRIUS.f90 kinds.o
	$(FC) $(FCFLAGS) ./model/US76_PATRIUS.f90

nrlmsise00_sub.o: ./model/nrlmsise00_sub.for
	$(FC) $(FCFLAGS) -fdefault-real-8 -std=legacy -w ./model/nrlmsise00_sub.for

srp.o: ./model/srp.f90 phys_const.o kinds.o
	$(FC) $(FCFLAGS) ./model/srp.f90

perturbations.o: ./model/perturbations.f90 kinds.o nsgrav.o sun_moon.o \
drag_exponential.o srp.o US76_PATRIUS.o nrlmsise00_sub.o ja77.o phys_const.o \
auxiliaries.o
	$(FC) $(FCFLAGS) ./model/perturbations.f90

initialize.o: initialize.f90 kinds.o settings.o auxiliaries.o $(FORMUL) \
phys_const.o nsgrav.o
	$(FC) $(FCFLAGS) initialize.f90

integrate.o: ./integ/integrate.f90 kinds.o phys_const.o io.o $(SLSODAR) \
$(DLSODAR)
	$(FC) $(FCFLAGS) ./integ/integrate.f90

propagate.o: propagate.f90 kinds.o settings.o auxiliaries.o phys_const.o \
$(FORMUL) integrate.o initialize.o io.o
	$(FC) $(FCFLAGS) propagate.f90

thalassa_main.o: thalassa_main.f90 kinds.o io.o phys_const.o settings.o \
sun_moon.o cart_coe.o propagate.o
	$(FC) $(FCFLAGS) thalassa_main.f90

# Atmospheric model
atsu.o: ./model/J77/atsu.for
	$(FC) $(FCFLAGS) -fdefault-real-8 -std=legacy ./model/J77/atsu.for

soflud.o: ./model/J77/soflud.for
	$(FC) $(FCFLAGS) -fdefault-real-8 -std=legacy ./model/J77/soflud.for

ja77.o: ./model/J77/ja77.for atsu.o soflud.o
	$(FC) $(FCFLAGS) -fdefault-real-8 -std=legacy ./model/J77/ja77.for

# Integrators
# SLSODAR
opksa1.o: ./integ/SLSODAR/opksa1.f kinds.o
	$(FC) $(FCFLAGS) -fdefault-real-8 -std=legacy ./integ/SLSODAR/opksa1.f

opksa2.o: ./integ/SLSODAR/opksa2.f kinds.o
	$(FC) $(FCFLAGS) -fdefault-real-8 -std=legacy ./integ/SLSODAR/opksa2.f

opksmain.o: ./integ/SLSODAR/opksmain.f kinds.o
	$(FC) $(FCFLAGS) -fdefault-real-8 -std=legacy ./integ/SLSODAR/opksmain.f

# DLSODAR
opkda1.o: ./integ/DLSODAR/opkda1.f kinds.o
	$(FC) $(FCFLAGS) -fdefault-real-8 -std=legacy ./integ/DLSODAR/opkda1.f

opkda2.o: ./integ/DLSODAR/opkda2.f kinds.o
	$(FC) $(FCFLAGS) -fdefault-real-8 -std=legacy ./integ/DLSODAR/opkda2.f

opkdmain.o: ./integ/DLSODAR/opkdmain.f kinds.o
	$(FC) $(FCFLAGS) -fdefault-real-8 -std=legacy ./integ/DLSODAR/opkdmain.f

# Formulations
regular_aux.o: ./regular/regular_aux.f90 perturbations.o edromo.o ks.o \
stische.o kinds.o
	$(FC) $(FCFLAGS) ./regular/regular_aux.f90

cowell.o: ./regular/cowell.f90 perturbations.o auxiliaries.o sun_moon.o kinds.o
	$(FC) $(FCFLAGS) ./regular/cowell.f90

edromo.o: ./regular/edromo.f90 perturbations.o settings.o auxiliaries.o \
nsgrav.o phys_const.o sun_moon.o kinds.o
	$(FC) $(FCFLAGS) ./regular/edromo.f90

ks.o: ./regular/ks.f90 perturbations.o settings.o auxiliaries.o phys_const.o \
nsgrav.o sun_moon.o kinds.o
	$(FC) $(FCFLAGS) ./regular/ks.f90

stische.o: ./regular/stische.f90 settings.o auxiliaries.o phys_const.o \
nsgrav.o perturbations.o sun_moon.o kinds.o
	$(FC) $(FCFLAGS) ./regular/stische.f90

.PHONY: clean
clean:
	-rm *.x *.o *.mod
