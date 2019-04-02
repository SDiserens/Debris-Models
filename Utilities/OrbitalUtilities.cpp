#include "stdafx.h"


uniform_real_distribution<double> uniformDistribution(0, 1);
double muGravity = GravitationalConstant * massEarth;
uint64_t seed = (uint64_t) chrono::system_clock::now().time_since_epoch().count();

//default_random_engine generator(seed);
//mt19937 generator(seed);
//mt19937_64 generator(seed);
mt19937_64 * generator = new mt19937_64(seed);

void SeedRNG(uint64_t seed)
{
	generator = new mt19937_64(seed);
}

/*
// Testing rand gen
static unsigned int x = 123456789, y = 362436000, z = 521288629, c = 7654321; // Seed variables
																			  
unsigned int KISS()
{
	unsigned long long t, a = 698769069ULL;
	x = 69069 * x + 12345;
	y ^= (y << 13);
	y ^= (y >> 17);
	y ^= (y << 5); // y must never be set to zero! 
	t = a*z + c;
	c = (t >> 32); // Also avoid setting z=c=0! 
	return x + y + (z = t);
}

double randomNumber()
{
	double x;
	unsigned int a, b;
	a = KISS() >> 6; // Upper 26 bits 
	b = KISS() >> 5; // Upper 27 bits 
	x = (a * 134217728.0 + b) / 9007199254740992.0;
	return x;
}
// test end 
*/


double randomNumber()
{
	return uniformDistribution(*generator);
}

double randomNumber(double max)
{ 
	return max * randomNumber();
}


double randomNumber(double min, double max)
{
	return min + (max - min) * randomNumber();
}

double randomNumberPi()
{
	return randomNumber(Pi);
}

double TauRange(double n)
{
	n = fmod(n, Tau);
	if (n < 0)
		n = Tau + n;

	return n;
}

double PiRange(double n)
{
	n = fmod(n, Pi);
	if (n < 0)
		n = Pi - n;
		
	return n;
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

vector3D CalculateAcceleration(vector3D & position)
{
	double rMagnitude = position.vectorNorm();
	return position * -muGravity / (rMagnitude * rMagnitude * rMagnitude);
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