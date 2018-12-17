#pragma once
extern std::default_random_engine generator;

const double massEarth = 5.972e24;  // Mass of earth in Kg
const double GravitationalConstant = 6.67408e-11;    // Gravitational constant(m ^ 3 kg^-1 s^-2)
const double muGravity = massEarth * GravitationalConstant;    // Combined for simplicity

class vector3D
{
public:
	double x, y, z;

public:
	vector3D();
	vector3D(double X, double Y, double Z);
	double vectorNorm(int ord);
	double vectorNorm();
	double vectorNorm2();
	void addVector(vector3D& vectorB);
	vector3D CalculateRelativeVector(vector3D& vectorB) const;
	vector3D operator+(vector3D& vectorB);
	vector3D operator-(vector3D& vectorB);
	vector3D operator/(double scalar);
	vector3D operator*(double scalar);
	double VectorDotProduct(vector3D& vectorB);
	vector3D VectorCrossProduct(vector3D& vectorB) const;

};

double CalculateKineticEnergy(vector3D& relativeVelocity, double mass);
double CalculateKineticEnergy(double speed, double mass);
double randomNumber();
double randomNumberTau();
double randomNumberPi();