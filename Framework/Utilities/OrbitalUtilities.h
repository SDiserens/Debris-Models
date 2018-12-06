#pragma once

class OrbitalElements
{
public:
	double semiMajorAxis, eccentricity, inclination, rightAscension, argPerigee;

public:
	OrbitalElements();
	OrbitalElements(double a, double e, double i, double ra, double ap);
};

class OrbitalAnomalies
{
public:
	bool synchronised; // Defines whether all the anomalies are currently up to date with each other.
protected:
	double meanAnomaly, eccentricAnomaly, trueAnomaly;
	int priority; // Defines which anomaly has most recently been set: 0, 1, 2 respectively.

public:
	void SetMeanAnomaly(double M);
	void SetTrueAnomaly(double v);
	void SetEccentricAnomaly(double E);
	double GetMeanAnomaly();
	double GetTrueAnomaly();
	double GetEccentricAnomaly();
};

class vector3D
{
public:
	double x, y, z;

public:
	vector3D();
	vector3D(double X, double Y, double Z);
	double vectorNorm(int ord);
	double vectorNorm2();
	vector3D CalculateRelativeVector(vector3D vectorB);
};

float CalculateKineticEnergy(vector3D relativeVelocity, float mass);
