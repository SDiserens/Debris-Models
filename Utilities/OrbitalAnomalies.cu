#include "stdafx.h"


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


double OrbitalAnomalies::EccentricToMeanAnomaly(double eA, double eccentricity)
{
	double tempAnomaly;
	tempAnomaly = eA - eccentricity * sin(eA);
	return TauRange(tempAnomaly);

}

double OrbitalAnomalies::EccentricToTrueAnomaly(double eA, double eccentricity)
{
	double X, Y, tempAnomaly;
	X = sqrt(1 - eccentricity) * cos(eA / 2);
	Y = sqrt(1 + eccentricity) * sin(eA / 2);
	tempAnomaly = 2 * atan2(Y, X);
	return tempAnomaly;
}

double OrbitalAnomalies::TrueToEccentricAnomaly(double tA, double eccentricity)
{
	double X, Y, tempAnomaly;
	Y = sqrt(1 - eccentricity * eccentricity) * sin(tA);
	X = eccentricity + cos(tA);
	tempAnomaly = atan2(Y, X);
	return tempAnomaly;
}

double OrbitalAnomalies::TrueToMeanAnomaly(double tA, double eccentricity)
{
	double tempAnomaly;
	tempAnomaly = TrueToEccentricAnomaly(tA, eccentricity);
	tempAnomaly = EccentricToMeanAnomaly(tempAnomaly, eccentricity);

	return tempAnomaly;
}

double OrbitalAnomalies::MeanToEccentricAnomaly(double mA, double eccentricity)
{
	int it = 0;
	double f, fPrime, M_new;
	double tempAnomaly = mA;
	double h = 1.0;
	if (eccentricity < 0.995) {
		while ((abs(h) >= NEWTONTOLERANCE) && (it < NEWTONMAXITERATIONS))
		{
			f = tempAnomaly - eccentricity * sin(tempAnomaly) - mA;
			fPrime = 1 - eccentricity * cos(tempAnomaly);

			h = f / fPrime;
			tempAnomaly -= h;

			++it;
		}
	}
	else {
		f = eccentricity / 2;
		tempAnomaly = mA + h;

		while (h > NEWTONTOLERANCE) {
			M_new = tempAnomaly - eccentricity*sin(tempAnomaly);
			h = mA - M_new;
			if (h > 0)
				tempAnomaly = tempAnomaly + f;
			else if (h < 0)
				tempAnomaly = tempAnomaly - f;
			else
				break;
			f = f / 2;
			++it;
		}
	}
	tempAnomaly = TauRange(tempAnomaly);
	return tempAnomaly;
}

double OrbitalAnomalies::MeanToTrueAnomaly(double mA, double eccentricity)
{
	double tempAnomaly;
	tempAnomaly = MeanToEccentricAnomaly(mA, eccentricity);
	tempAnomaly = EccentricToTrueAnomaly(tempAnomaly, eccentricity);

	return tempAnomaly;
}

void OrbitalAnomalies::UpdateAnomaliesFromEccentric(double eccentricity)
{
	meanAnomaly = EccentricToMeanAnomaly(eccentricAnomaly, eccentricity);
	trueAnomaly = EccentricToTrueAnomaly(eccentricAnomaly, eccentricity);
	synchronised = true;
}

void OrbitalAnomalies::UpdateAnomaliesFromTrue(double eccentricity)
{

	eccentricAnomaly = TrueToEccentricAnomaly(trueAnomaly, eccentricity);
	meanAnomaly = EccentricToMeanAnomaly(eccentricAnomaly, eccentricity);
	synchronised = true;
}

void OrbitalAnomalies::UpdateAnomaliesFromMean(double eccentricity)
{
	eccentricAnomaly = MeanToEccentricAnomaly(meanAnomaly, eccentricity);
	trueAnomaly = EccentricToTrueAnomaly(eccentricAnomaly, eccentricity);
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