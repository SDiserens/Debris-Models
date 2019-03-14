#pragma once
class OrbitalAnomalies
{
public:
	bool synchronised; // Defines whether all the anomalies are currently up to date with each other.
protected:
	double meanAnomaly, eccentricAnomaly, trueAnomaly;
	int priority; // Defines which anomaly has most recently been set: 0, 1, 2 respectively.

public:
	OrbitalAnomalies();
	~OrbitalAnomalies();
	void SetMeanAnomaly(double M);
	void SetTrueAnomaly(double v);
	void SetEccentricAnomaly(double E);
	double GetMeanAnomaly(double eccentricity);
	double GetTrueAnomaly(double eccentricity);
	double GetEccentricAnomaly(double eccentricity);

	double EccentricToMeanAnomaly(double eA, double eccentricity);
	double EccentricToTrueAnomaly(double eA, double eccentricity);
	double TrueToEccentricAnomaly(double tA, double eccentricity);
	double TrueToMeanAnomaly(double tA, double eccentricity);
	double MeanToEccentricAnomaly(double mA, double eccentricity);
	double MeanToTrueAnomaly(double mA, double eccentricity);

protected:
	void UpdateAnomaliesFromEccentric(double eccentricity);
	void UpdateAnomaliesFromMean(double eccentricity);
	void UpdateAnomaliesFromTrue(double eccentricity);

};

