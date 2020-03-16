#pragma once
extern mt19937_64 * generator;
//extern mt19937 * generator;

const int NEWTONMAXITERATIONS = 50;
const double NEWTONTOLERANCE = 1e-10;

CUDA_CONSTANT const double Pi = _Pi;
CUDA_CONSTANT const double Tau = 2 * _Pi;
const double secondsDay = 24 * 60 * 60;
const double secondsYear = secondsDay * 365;

const double massSol = 1.989e30;  // Mass of sun in Kg
const double massEarth = 5.972e24;  // Mass of earth in Kg
const double massJupiter = 1.898e27;  // Mass of jupiter in Kg

const double radiusEarth = 6378137; // in m
const double GravitationalConstant = 6.67408e-20;    // Gravitational constant(km ^ 3 kg^-1 s^-2)

extern double muGravity;    // Combined for simplicity

//unsigned int KISS();

class NewtonConvergenceException : public exception
{
	virtual const char* what() const throw()
	{
		return "Newton Method failed to converge";
	}
} ;

class ProgressBar
{
public:
	double n;
	char display;
	double percent;
	uint64_t pInt;
	string outputString;

	ProgressBar(double n, char d) : n(n/100), display(d) {}

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
	CUDA_CALLABLE_MEMBER vector3D CalculateRelativeVector(vector3D& vectorB) const;
	CUDA_CALLABLE_MEMBER vector3D operator+(vector3D& vectorB);
	CUDA_CALLABLE_MEMBER vector3D operator-(vector3D& vectorB);
	CUDA_CALLABLE_MEMBER vector3D operator/(double scalar);
	CUDA_CALLABLE_MEMBER vector3D operator*(double scalar);
	CUDA_CALLABLE_MEMBER double VectorDotProduct(vector3D& vectorB);
	CUDA_CALLABLE_MEMBER vector3D VectorCrossProduct(vector3D& vectorB) const;

};


void SeedRNG(uint64_t seed);
void SetCentralBody(int centralBody = 3);

vector3D CalculateAcceleration(vector3D& position);
double CalculateKineticEnergy(vector3D& relativeVelocity, double mass);
double CalculateKineticEnergy(double speed, double mass);
double DegToRad(double angle);
double RadtoDeg(double angle);
double randomNumber();
double randomNumber(double max);
double randomNumber(double min, double max);
double randomNumberTau();
double randomNumberPi();
double TauRange(double n);
double PiRange(double n);

double DateToEpoch(int year, int month, int day, int hour, int minute, double second);
double DateToEpoch(string date);