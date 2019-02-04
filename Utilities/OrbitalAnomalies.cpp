#include "stdafx.h"
#include "OrbitalAnomalies.h"


OrbitalAnomalies::OrbitalAnomalies()
{
}

OrbitalAnomalies::~OrbitalAnomalies()
{
}

// Anomalies Functions
void OrbitalAnomalies::SetMeanAnomaly(double M)
{
	meanAnomaly = M;
	synchronised = false;
	priority = 0;
}

void OrbitalAnomalies::SetTrueAnomaly(double v)
{
	trueAnomaly = v;
	synchronised = false;
	priority = 2;
}

void OrbitalAnomalies::SetEccentricAnomaly(double E)
{
	eccentricAnomaly = E;
	synchronised = false;
	priority = 1;
}


double OrbitalAnomalies::EccentricToMeanAnomaly(double eccentricity)
{
	double tempAnomaly;
	tempAnomaly = eccentricAnomaly - eccentricity * sin(eccentricAnomaly);
	return fmod(tempAnomaly, Tau);

}

double OrbitalAnomalies::EccentricToTrueAnomaly(double eccentricity)
{
	double X, Y, tempAnomaly;
	X = sqrt(1 - eccentricity) * cos(eccentricAnomaly / 2);
	Y = sqrt(1 + eccentricity) * sin(eccentricAnomaly / 2);
	tempAnomaly = 2 * atan2(Y, X);
	return tempAnomaly;
}

double OrbitalAnomalies::TrueToEccentricAnomaly(double eccentricity)
{
	double X, Y, tempAnomaly;
	Y = sqrt(1 - eccentricity * eccentricity) * sin(trueAnomaly);
	X = eccentricity + cos(trueAnomaly);
	tempAnomaly = atan2(Y, X);
	return tempAnomaly;
}

double OrbitalAnomalies::MeanToEccentricAnomaly(double eccentricity)
{
	int it = 0;
	double f, fPrime;
	double tempAnomaly = meanAnomaly;
	double h = 1.0;
	while ((abs(h) >= NEWTONTOLERANCE) && (it < NEWTONMAXITERATIONS))
	{
		f = tempAnomaly - eccentricity * sin(tempAnomaly) - meanAnomaly;
		fPrime = 1 - eccentricity * cos(tempAnomaly);

		h = f / fPrime;
		tempAnomaly -= h;

		it++;
	}
	return fmod(tempAnomaly, Tau);
}

void OrbitalAnomalies::UpdateAnomaliesFromEccentric(double eccentricity)
{
	meanAnomaly = EccentricToMeanAnomaly(eccentricity);
	trueAnomaly = EccentricToTrueAnomaly(eccentricity);
	synchronised = true;
}

void OrbitalAnomalies::UpdateAnomaliesFromTrue(double eccentricity)
{
	
	eccentricAnomaly = TrueToEccentricAnomaly(eccentricity);
	meanAnomaly = EccentricToMeanAnomaly(eccentricity);
	synchronised = true;
}

void OrbitalAnomalies::UpdateAnomaliesFromMean(double eccentricity)
{
	eccentricAnomaly = MeanToEccentricAnomaly(eccentricity);
	trueAnomaly = EccentricToTrueAnomaly(eccentricity);
	synchronised = true;
}

double OrbitalAnomalies::GetMeanAnomaly(double eccentricity)
{
	if (!synchronised)
		if (priority == 1)
			UpdateAnomaliesFromEccentric(eccentricity);
		else if (priority == 2)
			UpdateAnomaliesFromTrue(eccentricity);

	return meanAnomaly;
}

double OrbitalAnomalies::GetTrueAnomaly(double eccentricity)
{
	if (!synchronised)
		if (priority == 1)
			UpdateAnomaliesFromEccentric(eccentricity);
		else if (priority == 0)
			UpdateAnomaliesFromMean(eccentricity);

	return trueAnomaly;
}

double OrbitalAnomalies::GetEccentricAnomaly(double eccentricity)
{
	if (!synchronised)
		if (priority == 0)
			UpdateAnomaliesFromMean(eccentricity);
		else if (priority == 2)
			UpdateAnomaliesFromTrue(eccentricity);

	return eccentricAnomaly;
}