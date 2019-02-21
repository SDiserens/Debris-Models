// OrbitTrace.cpp : contains the implementation of the Orbit Trace collision algorithm.
//

#include "stdafx.h"
#include "OrbitTrace.h"


void OrbitTrace::MainCollision(DebrisPopulation& population, double timeStep)
{
}

double OrbitTrace::CollisionRate(DebrisObject & objectI, DebrisObject & objectJ)
{
	double collisionRate;
	vector3D velocityI, velocityJ, relativeVelocity;

	velocityI = objectI.GetVelocity();
	velocityJ = objectJ.GetVelocity();

	relativeVelocity = velocityI.CalculateRelativeVector(velocityJ);

	//TODO - OT collision rate

	return collisionRate;
}


bool OrbitTrace::CoplanarFilter(CollisionPair objectPair)
{
	// TODO -OT Coplanar filter
	return false;
}

bool OrbitTrace::HeadOnFilter(CollisionPair objectPair)
{
	// TODO - OT Head on filter
	return false;
}

bool OrbitTrace::SynchronizedFilter(CollisionPair objectPair)
{
	// TODO - OT synch filter
	return false;
}

bool OrbitTrace::ProximityFilter(CollisionPair objectPair)
{
	// TODO - OT  proximity filter
	return false;
}

/*
double OrbitTrace::CalculateSpatialDensity(DebrisObject object, double radius, double latitude)
{
	// Equation 21
	return 0.0;
}

double OrbitTrace::CalculateRadialSpatialDensity(DebrisObject object, double radius)
{
	// Equation 8A
	return 0.0;
}

double OrbitTrace::CalculateLatitudinalSpatialDensityRatio(DebrisObject object, double latitude)
{
	// Equations 13A
	return 0.0;
}

double OrbitTrace::CalculateVolumeElement(double radius, double latitude)
{
	// Equation 17
	return 0.0;
}
*/

