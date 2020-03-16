#pragma once
class OrbitalAnomalies
{
public:
	bool synchronised; // Defines whether all the anomalies are currently up to date with each other.
protected:
	double meanAnomaly, eccentricAnomaly, trueAnomaly;
	int priority; // Defines which anomaly has most recently been set: 0, 1, 2 respectively.

public:
	CUDA_CALLABLE_MEMBER OrbitalAnomalies();
	CUDA_CALLABLE_MEMBER ~OrbitalAnomalies();
	CUDA_CALLABLE_MEMBER void SetMeanAnomaly(double M);
	CUDA_CALLABLE_MEMBER void SetTrueAnomaly(double v);
	CUDA_CALLABLE_MEMBER void SetEccentricAnomaly(double E);
	CUDA_CALLABLE_MEMBER double GetMeanAnomaly(double eccentricity);
	CUDA_CALLABLE_MEMBER double GetTrueAnomaly(double eccentricity);
	CUDA_CALLABLE_MEMBER double GetEccentricAnomaly(double eccentricity);

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

