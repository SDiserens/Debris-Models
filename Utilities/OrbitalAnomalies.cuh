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

	CUDA_CALLABLE_MEMBER double EccentricToMeanAnomaly(double eA, double eccentricity);
	CUDA_CALLABLE_MEMBER double EccentricToTrueAnomaly(double eA, double eccentricity);
	CUDA_CALLABLE_MEMBER double TrueToEccentricAnomaly(double tA, double eccentricity);
	CUDA_CALLABLE_MEMBER double TrueToMeanAnomaly(double tA, double eccentricity);
	CUDA_CALLABLE_MEMBER double MeanToEccentricAnomaly(double mA, double eccentricity);
	CUDA_CALLABLE_MEMBER double MeanToTrueAnomaly(double mA, double eccentricity);

protected:
	CUDA_CALLABLE_MEMBER void UpdateAnomaliesFromEccentric(double eccentricity);
	CUDA_CALLABLE_MEMBER void UpdateAnomaliesFromMean(double eccentricity);
	CUDA_CALLABLE_MEMBER void UpdateAnomaliesFromTrue(double eccentricity);

};

