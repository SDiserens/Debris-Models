// OrbitTrace.cpp : contains the implementation of the Orbit Trace collision algorithm.
//

#include "stdafx.h"
#include "OrbitTrace.h"


void OrbitTrace::MainCollision(DebrisPopulation& population, double timeStep)
{
}

double OrbitTrace::CollisionRate(DebrisObject & objectI, DebrisObject & objectJ)
{
	double collisionCrossSection;
	vector3D velocityI, velocityJ, relativeVelocity;

	velocityI = objectI.GetVelocity();
	velocityJ = objectJ.GetVelocity();

	relativeVelocity = velocityI.CalculateRelativeVector(velocityJ);

	collisionCrossSection = CollisionCrossSection(objectI, objectJ);

	return collisionCrossSection * relativeVelocity.vectorNorm();
}

vector<pair<long, long>> OrbitTrace::CreatePairList(DebrisPopulation & population)
{
	return vector<pair<long, long>>();
}

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

