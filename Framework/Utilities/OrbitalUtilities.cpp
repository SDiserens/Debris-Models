#include "stdafx.h"

float CalculateKineticEnergy(vector3D relativeVelocity, float mass)
{
	float kineticEnergy = 0.5 * mass * relativeVelocity.vectorNorm2();
	return kineticEnergy;
}

// 3D vector functions
vector3D::vector3D() {}

vector3D::vector3D(double X, double Y, double Z)
{
	x = X;
	y = Y;
	z = Z;
}

double vector3D::vectorNorm(int ord = 2)
{
	double norm;
	norm = sqrt(pow(x, ord) + pow(y, ord) + pow(z, ord));
	return norm;
}

double vector3D::vectorNorm2()
{
	double norm;
	norm = x * x + y * y + z * z;
	return norm;
}

vector3D vector3D::CalculateRelativeVector(vector3D vectorB)
{
	double X, Y, Z;
	X = x - vectorB.x;
	Y = y - vectorB.y;
	Z = z - vectorB.z;
	return vector3D(X, Y, Z);
}

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

// Anomalies Functions
void OrbitalAnomalies::SetMeanAnomaly(double M)
{
	meanAnomaly = M;
	synchronised = false;
	priority = 0;
}

void OrbitalAnomalies::SetTrueAnomaly(double v)
{
	meanAnomaly = v;
	synchronised = false;
	priority = 2;
}

void OrbitalAnomalies::SetEccentricAnomaly(double E)
{
	meanAnomaly = E;
	synchronised = false;
	priority = 1;
}

double OrbitalAnomalies::GetMeanAnomaly()
{
	return 0.0;
}

double OrbitalAnomalies::GetTrueAnomaly()
{
	return 0.0;
}

double OrbitalAnomalies::GetEccentricAnomaly()
{
	return 0.0;
}
