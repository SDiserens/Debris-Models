#include "stdafx.h"

#include <chrono>

uniform_real_distribution<double> uniformDistribution(0, 1);

uint64_t seed = (uint64_t) chrono::system_clock::now().time_since_epoch().count();

//default_random_engine generator(seed);
//mt19937 generator(seed);
//mt19937_64 generator(seed);
mt19937 * generator = new mt19937(seed);


void SeedRNG(uint64_t seed)
{
	generator = new mt19937(seed);
}

double muGravity = GravitationalConstant * massEarth;

double randomNumber()
{
	return uniformDistribution(*generator);
}

double randomNumber(double max)
{
	return max * uniformDistribution(*generator);
}


double randomNumber(double min, double max)
{
	return min + (max - min) * uniformDistribution(*generator);
}

double randomNumberPi()
{
	return randomNumber(Pi);
}

double randomNumberTau()
{
	return randomNumber(Tau);
}

void SetCentralBody(int centralBody)
{
	double centralMass;
	if (centralBody == 0)
		centralMass = massSol;
	else if (centralBody == 3)
		centralMass = massEarth;
	else if (centralBody == 5)
		centralMass = massJupiter;


	muGravity = centralMass * GravitationalConstant;
}

double CalculateKineticEnergy(vector3D& relativeVelocity, double mass)
{
	double kineticEnergy = 0.5 * mass * relativeVelocity.vectorNorm2() * 1e6;
	return kineticEnergy;
}

double CalculateKineticEnergy(double speed, double mass)
{
	speed *= 1000; // Convert to M/s
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