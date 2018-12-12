#include "stdafx.h"

int objectSEQ = 0;

DebrisObject::DebrisObject() {}

DebrisObject::DebrisObject(double init_radius, double init_mass, double init_length, double semiMajorAxis, double eccentricity, double inclination,
	double rightAscension, double argPerigee, double init_meanAnomaly)
{
	objectID = ++objectSEQ; // This should be generating a unique ID per object incl. when threading
	radius = init_radius;
	mass = init_mass;
	length = init_length;
	OrbitalElements elements(semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee);
	OrbitalAnomalies anomalies;
	meanAnomalyEpoch = init_meanAnomaly;
	anomalies.SetMeanAnomaly(init_meanAnomaly);
}



DebrisObject::~DebrisObject()
{
}

void DebrisObject::UpdateOrbitalElements(double deltaV)
{

}

vector3D DebrisObject::GetVelocity()
{
	return velocity;
}

void DebrisObject::SetVelocity()
{
	velocity = vector3D(0.0, 0.0, 0.0);
}

void DebrisObject::CalculateMassFromArea()
{
	mass = area / areaToMass;
}


void DebrisObject::CalculateAreaFromMass()
{
	area = mass * areaToMass;
}

void DebrisObject::CalculateAreaToMass()
{
	areaToMass = area / mass;
}

double DebrisObject::GetMass()
{
	return mass;
}

double DebrisObject::GetLength()
{
	return length;
}

double DebrisObject::GetArea()
{
	return area;
}

double DebrisObject::GetAreaToMass()
{
	return areaToMass;
}

