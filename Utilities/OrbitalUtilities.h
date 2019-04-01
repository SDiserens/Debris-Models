#pragma once
extern mt19937_64 * generator;
//extern mt19937 * generator;

const double NEWTONMAXITERATIONS = 20;
const double NEWTONTOLERANCE = 1e-13;

const double Pi = _Pi;
const double Tau = 2 * _Pi;
const double secondsDay = 24 * 60 * 60;
const double secondsYear = secondsDay * 365;

const double massSol = 1.989e30;  // Mass of sun in Kg
const double massEarth = 5.972e24;  // Mass of earth in Kg
const double massJupiter = 1.898e27;  // Mass of jupiter in Kg

const double GravitationalConstant = 6.67408e-20;    // Gravitational constant(km ^ 3 kg^-1 s^-2)

extern double muGravity;    // Combined for simplicity

//unsigned int KISS();

class ProgressBar
{
public:
	uint64_t n;
	char display;
	double percent;
	uint64_t pInt;

	ProgressBar(uint64_t n, char d) : n(n/100), display(d) {}

	void DisplayProgress(uint64_t i)
	{
		/* A function for displaying a progress bar.
		*/
		if (i == 0)
		{
			cout << fixed << setprecision(2) << showpoint;
			cout << '\r' + string(100, ' ') + ": " << 0.00 << "%" << flush;
		}
		if ((i % (n / 100)) == 0)
		{
			percent = ((double)i / n);
			pInt = uint64_t(percent);
			cout << fixed << setprecision(2) << showpoint;
			cout << '\r' + string(pInt, display) + string((100 - pInt), ' ') + ": " << percent << "%" << flush;
		}
	}
};

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


void SeedRNG(uint64_t seed);
void SetCentralBody(int centralBody = 3);

vector3D CalculateAcceleration(vector3D& position);
double CalculateKineticEnergy(vector3D& relativeVelocity, double mass);
double CalculateKineticEnergy(double speed, double mass);
double randomNumber();
double randomNumber(double max);
double randomNumber(double min, double max);
double randomNumberTau();
double randomNumberPi();
double TauRange(double n);
double PiRange(double n);