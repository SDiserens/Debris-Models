#pragma once
class OrbitalElements
{
public:
	double semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee;

public:
	OrbitalElements();
	OrbitalElements(double a, double e, double i, double ra, double ap);
	OrbitalElements(vector3D position, vector3D velocity);
};