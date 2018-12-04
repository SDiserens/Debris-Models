#include "stdafx.h"
#include <vector>

int objectSEQ = 0;

DebrisObject::DebrisObject(float init_radius, float init_mass, float init_length, double semiMajorAxis, double eccentricity, double inclination,
	double rightAscension, double argPerigee, double init_meanAnomaly) : elements(5), anomalies(3)
{
	objectID = ++objectSEQ; // This should be generating a unique ID per object incl. when threading
	radius = init_radius;
	mass = init_mass;
	length = init_length;
	elements[0] = semiMajorAxis;
	elements[1] = eccentricity;
	elements[2] = inclination;
	elements[3] = rightAscension;
	elements[4] = argPerigee;
	meanAnomalyEpoch = anomalies[0] = init_meanAnomaly;
}



DebrisObject::~DebrisObject()
{
}

void DebrisObject::UpdateOrbitalElements(double deltaV)
{

}