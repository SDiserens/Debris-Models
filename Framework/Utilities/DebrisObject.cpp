#include "stdafx.h"

int objectSEQ = 0;

DebrisObject::DebrisObject() {}

DebrisObject::DebrisObject(float init_radius, float init_mass, float init_length, double semiMajorAxis, double eccentricity, double inclination,
	double rightAscension, double argPerigee, double init_meanAnomaly)
{
	objectID = ++objectSEQ; // This should be generating a unique ID per object incl. when threading
	radius = init_radius;
	mass = init_mass;
	length = init_length;
	OrbitalElements elements(semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee);
	OrbitalAnomalies anomalies;
	meanAnomalyEpoch = anomalies.meanAnomaly = init_meanAnomaly;
}



DebrisObject::~DebrisObject()
{
}

void DebrisObject::UpdateOrbitalElements(double deltaV)
{

}

vector3D DebrisObject::GetVelocity()
{
	vector3D velocity(0.0, 0.0, 0.0);
	return velocity;
}