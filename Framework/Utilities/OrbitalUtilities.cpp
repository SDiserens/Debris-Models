#include "stdafx.h"

double CalculateKineticEnergy(vector3D relativeVelocity, double mass)
{
	double kineticEnergy = 0.5 * mass * relativeVelocity.vectorNorm2();
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

