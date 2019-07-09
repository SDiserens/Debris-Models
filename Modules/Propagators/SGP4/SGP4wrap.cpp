#include "stdafx.h"
#include "SGP4wrap.h"


SGP4::SGP4()
{
}


SGP4::~SGP4()
{
}

void SGP4::UpdateElements(DebrisObject &object, double timeStep)
{
	double r[3];
	double v[3];
	// If first instance then call initialise SGP4 for object
	
		// Calculate BStar value

		// call SGP4init


	// Propagate forward by timestep in minutes

		// call SGP4 procedure

	// Update orbital elements
	object.UpdateOrbitalElements(vector3D(r[0], r[1], r[2]), vector3D(v[0], v[1], v[2]));
}
