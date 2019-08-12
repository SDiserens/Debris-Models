#include "stdafx.h"
#include "SGP4wrap.h"


SGP4::SGP4(DebrisPopulation & initPopulation) : Propagator(initPopulation)
{
}

SGP4::SGP4(DebrisPopulation & initPopulation, char opMode, gravconsttype gravMode) : Propagator(initPopulation)
{
	opsMode = opMode;
	gravModel = gravMode;
}

SGP4::~SGP4()
{
}

void SGP4::UpdateElements(DebrisObject &object, double timeStep)
{
	int orbitState;
	double bStar, timeMinutes, meanMotionKozai;
	double r[3];
	double v[3];
	// If first instance then call initialise SGP4 for object
	if (!object.SGP4Initialised())
	{

		OrbitalElements& elements = object.GetElements();
		// Calculate BStar value
		bStar = object.GetBStar();
		if (isnan(bStar))
		{
			bStar = CalculateBStar(object);
			object.SetBStar(bStar);
		}
		meanMotionKozai = Tau * elements.GetMeanMotion() / 1440; // rad/min
		// call SGP4init
		SGP4Funcs::sgp4init(gravModel, opsMode, object.GetID(), object.GetInitEpoch(), bStar, 0, 0, //first and second derivative of the mean motion set to zero as unused
										elements.eccentricity, elements.argPerigee, elements.inclination, elements.GetMeanAnomaly(), meanMotionKozai, elements.rightAscension,
										object.GetSGP4SatRec());

		if (object.GetSGP4SatRec().error != 0)
			HandleError(object);
	}

	// Propagate forward by timestep in minutes
	timeMinutes = (timeStep + population.GetEpoch()) * 24 * 60;
		// call SGP4 procedure
	orbitState = SGP4Funcs::sgp4(object.GetSGP4SatRec(), timeMinutes, r, v);
	
	if (!orbitState)
		HandleError(object);

	// Update orbital elements
	else
		object.SetStateVectors(r[0], r[1], r[2], v[0], v[1], v[2]);
}

void SGP4::HandleError(DebrisObject &object)
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

	if (errorCode == 6 || errorCode == 5 || errorCode == 2)
	{
		removeID.push_back(object.GetID());
	}
	else if (errorCode == 1 || errorCode == 3 || errorCode == 4)
	{
		removeID.push_back(object.GetID());
		// Todo add handling for non elliptical orbtis
	}
}
