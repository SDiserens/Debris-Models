#pragma once
class OrbitalElements
{
public:
	double semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee;

public:
	OrbitalElements();
	OrbitalElements(double a, double e, double i, double ra, double ap);
	OrbitalElements(vector3D &position, vector3D &velocity);
	double GetTrueAnomaly();
	void SetOrbitalElements(double a, double e, double i, double ra, double ap);
	void SetOrbitalElements(vector3D &position, vector3D &velocity);

private:
	double trueAnomaly;
	vector3D CalculateEccentricityVector(vector3D &position, vector3D &velocity, vector3D &angularMomentum);
};