#pragma once
extern std::default_random_engine generator;

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

class ProgressBar
{
public:
	int n;
	char display;
	float percent;
	unsigned long pInt;

	ProgressBar(int n, char d) : n(n), display(d) {}

	void DisplayProgress(unsigned long i)
	{
		/* A function for displaying a progress bar.
		*/
		if ((i % (n / 2000)) == 0)
		{
			percent = 100 * (i / float(n));
			pInt = unsigned long(percent);
			cout << '\r' + string(pInt, display) + string((100 - pInt), ' ') + ": " + to_string(percent) + "%." << flush;
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

void SetCentralBody(int centralBody = 3);

double CalculateKineticEnergy(vector3D& relativeVelocity, double mass);
double CalculateKineticEnergy(double speed, double mass);
double randomNumber();
double randomNumber(double max);
double randomNumber(double min, double max);
double randomNumberTau();
double randomNumberPi();