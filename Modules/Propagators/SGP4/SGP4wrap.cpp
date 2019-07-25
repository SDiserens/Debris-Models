#include "stdafx.h"
#include "SGP4wrap.h"


SGP4::SGP4(DebrisPopulation & initPopulation) : Propagator(initPopulation)
{
}


SGP4::~SGP4()
{
}

void SGP4::UpdateElements(DebrisObject &object, double timeStep)
{
	int orbitState;
	double bStar, timeMinutes;
	double r[3];
	double v[3];
	// If first instance then call initialise SGP4 for object
	if (object.SGP4Initialised())
	{

		OrbitalElements& elements = object.GetElements();
		// Calculate BStar value
		bStar = object.GetBStar();
		if (isnan(bStar))
		{
			bStar = CalculateBStar(object);
			object.SetBStar(bStar);
		}

		// call SGP4init
		orbitState = SGP4Funcs::sgp4init(gravModel, opsMode, object.GetID(), object.GetInitEpoch(), bStar, 0, 0, //first and second derivative of the mean motion set to zero as unused
										elements.eccentricity, elements.argPerigee, elements.inclination, elements.GetMeanAnomaly(), elements.GetMeanMotion(), elements.rightAscension,			
										object.GetSGP4SatRec());

		if (!orbitState)
			HandleSPG4Error(object);
	}

	// Propagate forward by timestep in minutes
	timeMinutes = timeStep * 24 * 60;
		// call SGP4 procedure
	orbitState = SGP4Funcs::sgp4(object.GetSGP4SatRec(), timeMinutes, r, v);
	
	if (!orbitState)
		HandleSPG4Error(object);

	// Update orbital elements
	else
		object.SetStateVectors(r[0], r[1], r[2], v[0], v[1], v[2]);
}

double SGP4::CalculateBStar(DebrisObject & object)
{
	double ballisticC;

	ballisticC = object.GetCDrag() * object.GetAreaToMass();

	return rhoZero / 2 * ballisticC;
}

void SGP4::HandleSPG4Error(DebrisObject &object)
{
	int errorCode;
	errorCode = object.GetSGP4SatRec().error;
	/*	
	*  return code - non-zero on error.
	*                   1 - mean elements, ecc >= 1.0 or ecc < -0.001 or a < 0.95 er
	*                   2 - mean motion less than 0.0
	*                   3 - pert elements, ecc < 0.0  or  ecc > 1.0
	*                   4 - semi-latus rectum < 0.0
	*                   5 - epoch elements are sub-orbital
	*                   6 - satellite has decayed
	*/

	//TODO - handle SGP4 error codes
	if (errorCode == 6)
	{
		population.DecayObject(object.GetID());
	}
	else if (errorCode == 5)
	{
		population.DecayObject(object.GetID());
	}
}
