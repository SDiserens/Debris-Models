#include "stdafx.h"
#include "OrbitalAnomalies.h"


// Anomalies Functions
void OrbitalAnomalies::SetMeanAnomaly(double M)
{
	meanAnomaly = M;
	synchronised = false;
	priority = 0;
}

void OrbitalAnomalies::SetTrueAnomaly(double v)
{
	meanAnomaly = v;
	synchronised = false;
	priority = 2;
}

void OrbitalAnomalies::SetEccentricAnomaly(double E)
{
	meanAnomaly = E;
	synchronised = false;
	priority = 1;
}

double OrbitalAnomalies::GetMeanAnomaly()
{
	return 0.0;
}

double OrbitalAnomalies::GetTrueAnomaly()
{
	return 0.0;
}

double OrbitalAnomalies::GetEccentricAnomaly()
{
	return 0.0;
}