#pragma once
class OrbitalAnomalies
{
public:
	bool synchronised; // Defines whether all the anomalies are currently up to date with each other.
protected:
	double meanAnomaly, eccentricAnomaly, trueAnomaly;
	int priority; // Defines which anomaly has most recently been set: 0, 1, 2 respectively.

public:
	void SetMeanAnomaly(double M);
	void SetTrueAnomaly(double v);
	void SetEccentricAnomaly(double E);
	double GetMeanAnomaly();
	double GetTrueAnomaly();
	double GetEccentricAnomaly();
};

