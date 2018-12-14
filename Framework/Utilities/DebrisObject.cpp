#include "stdafx.h"

int DebrisObject::objectSEQ = 0;

DebrisObject::DebrisObject() {}

DebrisObject::DebrisObject(double init_radius, double init_mass, double init_length, double semiMajorAxis, double eccentricity, double inclination,
	double rightAscension, double argPerigee, double init_meanAnomaly)
{
	objectID = ++objectSEQ; // This should be generating a unique ID per object incl. when threading (needs revision for multi-thread)
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

void DebrisObject::UpdateOrbitalElements(vector3D deltaV)
{
	velocity.addVector(deltaV);
	elements = OrbitalElements(position, velocity);
	anomalies.SetTrueAnomaly(elements.GetTrueAnomaly());
}

vector3D DebrisObject::GetVelocity()
{
	return velocity;
}

void DebrisObject::SetVelocity(double vX, double vY, double vZ)
{
	velocity = vector3D(vX, vY, vZ);
}

void DebrisObject::SetVelocity(vector3D inputVelocity)
{
	velocity = vector3D(inputVelocity.x, inputVelocity.y, inputVelocity.z);
}

vector3D DebrisObject::GetPosition()
{
	return position;
}

void DebrisObject::SetPosition(double X, double Y, double Z)
{
	velocity = vector3D(X, Y, Z);
}

void DebrisObject::SetPosition(vector3D inputPosition)
{
	velocity = vector3D(inputPosition.x, inputPosition.y, inputPosition.z);
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

