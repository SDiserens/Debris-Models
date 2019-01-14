#pragma once
extern std::default_random_engine generator;

const double NEWTONMAXITERATIONS = 20;
const double NEWTONTOLERANCE = 1e-13;

const double Pi = _Pi;
const double Tau = 2 * _Pi;

const double massEarth = 5.972e24;  // Mass of earth in Kg
const double GravitationalConstant = 6.67408e-20;    // Gravitational constant(km ^ 3 kg^-1 s^-2)
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
double randomNumber(double max);
double randomNumber(double min, double max);
double randomNumberTau();
double randomNumberPi();