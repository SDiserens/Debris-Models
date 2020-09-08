#pragma once
extern mt19937_64 * generator;
//extern mt19937 * generator;

#define NEWTONMAXITERATIONS 100
#define NEWTONTOLERANCE 1e-8

#define Pi _Pi
#define Tau (2*_Pi)
#define secondsDay 86400
#define secondsYear 31536000

#define massSol 1.989e30  // Mass of sun in Kg
#define massEarth 5.972e24  // Mass of earth in Kg
#define massJupiter 1.898e27  // Mass of jupiter in Kg

#define radiusEarth 6378137 // in m
#define GravitationalConstant 6.67408e-20    // Gravitational constant(km ^ 3 kg^-1 s^-2)

#define muGravity 398600.5
#define muSol 132747451200
#define muJov 126674038.4


//unsigned int KISS();

class NewtonConvergenceException : public exception
{
	virtual const char* what() const throw()
	{
		return "Newton Method failed to converge";
	}
};

class ProgressBar
{
public:
	double n;
	char display;
	double percent;
	uint64_t pInt;
	string outputString;

	ProgressBar(double n, char d) : n(n / 100), display(d) {}

	void DisplayProgress(double i)
	{
		/* A function for displaying a progress bar.
		*/
		if (i == 0)
		{
			//cout << fixed << setprecision(2) << showpoint;
			cout << '\r' + string(100, ' ') + ": " << 0.00 << "%" << flush;
		}
		else
		{
			percent = (i / n);
			pInt = uint64_t(percent);
			//cout << fixed << setprecision(2) << showpoint;
			outputString = '\r' + string(pInt, display) + string((100 - pInt), ' ') + ": %4.2f %%";
			printf(outputString.c_str(), percent);
			cout.flush();
		}
	}
};

class vector3D
{
public:
	double x, y, z;

public:
	CUDA_CALLABLE_MEMBER vector3D();
	CUDA_CALLABLE_MEMBER vector3D(double X, double Y, double Z);
	CUDA_CALLABLE_MEMBER double vectorNorm(int ord);
	CUDA_CALLABLE_MEMBER double vectorNorm();
	CUDA_CALLABLE_MEMBER double vectorNorm2();
	CUDA_CALLABLE_MEMBER void addVector(vector3D& vectorB);
	CUDA_CALLABLE_MEMBER vector3D CalculateRelativeVector(vector3D vectorB) const;
	CUDA_CALLABLE_MEMBER vector3D operator+(vector3D vectorB);
	CUDA_CALLABLE_MEMBER vector3D operator-(vector3D vectorB);
	CUDA_CALLABLE_MEMBER vector3D operator/(double scalar);
	CUDA_CALLABLE_MEMBER vector3D operator*(double scalar);
	CUDA_CALLABLE_MEMBER double VectorDotProduct(vector3D vectorB);
	CUDA_CALLABLE_MEMBER vector3D VectorCrossProduct(vector3D vectorB) const;

};


void SeedRNG(uint64_t seed);
//void SetCentralBody(int centralBody = 3);

double CalculateKineticEnergy(vector3D& relativeVelocity, double mass);
double CalculateKineticEnergy(double speed, double mass);
double DegToRad(double angle);
double RadtoDeg(double angle);
double randomNumber();
double randomNumber(double max);
double randomNumber(double min, double max);
double randomNumberTau();
double randomNumberPi();
CUDA_CALLABLE_MEMBER double TauRange(double n);
CUDA_CALLABLE_MEMBER double PiRange(double n);

double DateToEpoch(int year, int month, int day, int hour, int minute, double second);
double DateToEpoch(string date);
string EpochToDate(double epoch);