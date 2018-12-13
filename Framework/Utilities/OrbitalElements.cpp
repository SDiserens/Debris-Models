#include "stdafx.h"
#include "OrbitalElements.h"

// Elements functions
OrbitalElements::OrbitalElements() {}

OrbitalElements::OrbitalElements(double a, double e, double i, double ra, double ap)
{
	semiMajorAxis = a;
	eccentricity = e;
	inclination = i;
	rightAscension = ra;
	argPerigee = ap;
}


OrbitalElements::OrbitalElements(vector3D position, vector3D velocity)
{

}