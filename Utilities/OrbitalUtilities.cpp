#include "stdafx.h"

std::uniform_real_distribution<double> uniformDistribution(0, 1);
std::uniform_real_distribution<double> piDistribution(0, Pi);
std::uniform_real_distribution<double> tauDistribution(0, Tau);
std::default_random_engine generator;


double randomNumber()
{
	return uniformDistribution(generator);
}

double randomNumber(double max)
{
	return max * uniformDistribution(generator);
}


double randomNumber(double min, double max)
{
	return min + (max - min) * uniformDistribution(generator);
}

double randomNumberPi()
{
	return piDistribution(generator);
}

double randomNumberTau()
{
	return tauDistribution(generator);
}

double CalculateKineticEnergy(vector3D& relativeVelocity, double mass)
{
	double kineticEnergy = 0.5 * mass * relativeVelocity.vectorNorm2();
	return kineticEnergy;
}

double CalculateKineticEnergy(double speed, double mass)
{
	double kineticEnergy = 0.5 * mass * speed * speed;
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

double vector3D::vectorNorm(int ord)
{
	double norm;
	norm = sqrt(pow(x, ord) + pow(y, ord) + pow(z, ord));
	return norm;
}

double vector3D::vectorNorm()
{
	double norm;
	norm = sqrt(x * x + y * y + z * z);
	return norm;
}

double vector3D::vectorNorm2()
{
	double norm;
	norm = x * x + y * y + z * z;
	return norm;
}

vector3D vector3D::CalculateRelativeVector(vector3D& vectorB) const
{
	double X, Y, Z;
	X = x - vectorB.x;
	Y = y - vectorB.y;
	Z = z - vectorB.z;
	return vector3D(X, Y, Z);
}


double vector3D::VectorDotProduct(vector3D& vectorB)
{
	double product;
	product = x * vectorB.x + y * vectorB.y + vectorB.z;
	return product;
}

vector3D vector3D::VectorCrossProduct(vector3D& vectorB) const
{
	double X, Y, Z;
	X = y * vectorB.z - z * vectorB.y;
	Y = z * vectorB.x - x * vectorB.z;
	Z = x * vectorB.y - y * vectorB.x;
	return vector3D(X, Y, Z);
}

void vector3D::addVector(vector3D& vectorB)
{
	x += vectorB.x;
	y += vectorB.y;
	z += vectorB.z;
}

vector3D vector3D::operator+(vector3D& vectorB)
{
	double X, Y, Z;
	X = x + vectorB.x;
	Y = y + vectorB.y;
	Z = z + vectorB.z;
	return vector3D(X, Y, Z);
}

vector3D vector3D::operator-(vector3D& vectorB)
{
	double X, Y, Z;
	X = x - vectorB.x;
	Y = y - vectorB.y;
	Z = z - vectorB.z;
	return vector3D(X, Y, Z);
}


vector3D vector3D::operator/(double scalar)
{
	double X, Y, Z;
	X = x / scalar;
	Y = y / scalar;
	Z = z / scalar;
	return vector3D(X, Y, Z);
}

vector3D vector3D::operator*(double scalar)
{
	double X, Y, Z;
	X = x * scalar;
	Y = y * scalar;
	Z = z * scalar;
	return vector3D(X, Y, Z);
}